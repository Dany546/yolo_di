
from ultralytics.nn.modules.block import C2f, Bottleneck, Conv
from ultralytics.nn.modules.conv import autopad

# from .DI.CDFKD-MFS.nets import resnet_err

import torch.nn.functional as F
import torch

class BatchNorm2d(torch.nn.BatchNorm2d):
    def __init__(self, *args, **kwargs):
        super(BatchNorm2d, self).__init__(*args, **kwargs)
        self.reset_parameters()

    def reset_parameters(self) -> None: 
        self.reset_running_stats()
        if self.affine:
            torch.nn.init.zeros_(self.weight)
            torch.nn.init.zeros_(self.bias)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(input)
        # From torch documentation
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
 
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None) 
            
        return F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean
            if not self.training or self.track_running_stats
            else None,
            self.running_var if not self.training or self.track_running_stats else None,
            1 + self.weight,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )  

class ConvTransposed(torch.nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = torch.nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super(ConvTransposed, self).__init__()
        self.conv = torch.nn.ConvTranspose2d(c1, c2, k, s, padding=autopad(k,p,d), groups=g, dilation=d, bias=False)
        self.bn = BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, torch.nn.Module) else torch.nn.Identity()
        self.pad = lambda x: torch.nn.functional.pad(x, (1,1,1,1))
        self.crop = lambda x: x[:,:,1:-1,1:-1]

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.bn(self.act(self.crop(self.conv(self.pad(x)))))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.crop(self.conv(self.pad(x))))

class BottleneckTranspose(torch.nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super(BottleneckTranspose, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = ConvTransposed(c1, c_, k[0], 1)
        self.cv2 = ConvTransposed(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data.""" 
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2fTransposed(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(C2fTransposed, self).__init__(*args, **kwargs) 

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = ConvTransposed(c1, 2 * self.c, 1, 1)
        self.cv2 = ConvTransposed((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = torch.nn.ModuleList(BottleneckTranspose(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1)) 

class VectorQuantizerEMA(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay=0.9, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = torch.nn.Embedding(self._num_embeddings, self._embedding_dim).to(torch.device("cuda"))
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings) # normal_()
        self._commitment_cost = commitment_cost
        
        self.register_buffer('_ema_cluster_size', torch.ones(num_embeddings).to(torch.device("cuda")))
        self._ema_w = torch.nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim)).to(torch.device("cuda"))
        self._ema_w.data.normal_()
        
        self.utility = torch.ones(num_embeddings).to(torch.device("cuda"))/num_embeddings
        
        self._decay = decay
        self._epsilon = epsilon
        self.is_init = False

    def forward(self, inputs):
        # disable mixed precision for convergence issues
        inputs = inputs.float()
        with torch.cuda.amp.autocast(enabled=True):
            # convert inputs from BCHW -> BHWC
            inputs = inputs.permute(0, 2, 3, 1).contiguous()
            input_shape = inputs.shape
            
            # Flatten input
            flat_input = inputs.view(-1, self._embedding_dim)
            
            """
            if not self.training: # suppress the atoms of the dictionnary that were not used enough
                distances[self.utility<=threshold] = 
            """
            
            # Calculate distances
            distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                        + torch.sum(self._embedding.weight**2, dim=1)
                        - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
                
            # Encoding
            encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
            encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
            encodings.scatter_(1, encoding_indices, 1)
            
            # Quantize and unflatten
            quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        with torch.cuda.amp.autocast(enabled=False):
            # Use EMA to update the embedding vectors
            if self.training and not self.is_init: 
                self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                         (1 - self._decay) * torch.sum(encodings, 0)
                
                # Laplace smoothing of the cluster size
                n = torch.sum(self._ema_cluster_size.data)
                self._ema_cluster_size = (
                    (self._ema_cluster_size + self._epsilon)
                    / (n + self._num_embeddings * self._epsilon) * n)
                
                dw = torch.matmul(encodings.t(), flat_input)
                self._ema_w = torch.nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
                
                self._embedding.weight = torch.nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1)).to(torch.device("cuda")) 

        with torch.cuda.amp.autocast(enabled=True):
            # Loss
            e_latent_loss = torch.tensor([0.], device=inputs.device, requires_grad=self.training) 
            e_latent_loss = e_latent_loss + torch.mean(F.mse_loss(quantized.detach(), inputs, reduction='none'), dim=-1)
            if self.is_init: 
                q_latent_loss = F.mse_loss(quantized, inputs.detach())
                loss = q_latent_loss + self._commitment_cost * e_latent_loss
            else:
                loss = self._commitment_cost * e_latent_loss
            
            # Straight Through Estimator
            quantized = inputs + (quantized - inputs).detach()
            avg_probs = torch.mean(encodings, dim=0)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
            
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encoding_indices

class YOLODecoderBlock(torch.nn.Module):
    
    def __init__(self, c_in, c_out, *args, **kwargs):
        super(YOLODecoderBlock, self).__init__(*args, **kwargs) 
        
        # Define VQ layer
        self.vq_layer = VectorQuantizerEMA(256, c_in, 1, decay=0.9, epsilon=1e-5)
        # Define convolutional layers  
        c2f = C2fTransposed(c_in, c_out, n=2, shortcut=False)
        self.layers = torch.nn.ModuleList([torch.nn.ConvTranspose2d(c_in, c_out, 3, padding=1, bias=False), 
                                           torch.nn.SiLU(), 
                                           BatchNorm2d(c_out),
                                           torch.nn.ConvTranspose2d(c_out, c_out, 3, padding=1, bias=False), 
                                           torch.nn.SiLU(), 
                                           BatchNorm2d(c_out),
                                           torch.nn.Upsample(scale_factor=2, mode="nearest"), 
                                           torch.nn.ConvTranspose2d(c_out, c_out, 3, padding=1, bias=False), 
                                           torch.nn.SiLU(),
                                           BatchNorm2d(c_out),
                                           torch.nn.ConvTranspose2d(c_out, c_out, 3, padding=1, bias=False), 
                                           torch.nn.SiLU(), 
                                           BatchNorm2d(c_out),
                      ])
        
    def forward(self, input, warmup=1): 
        vq_loss, quantized, perplexity, encoding_indices = self.vq_layer(input)
        input = warmup*quantized + (1-warmup)*input
        for layer in self.layers:
            input = layer(input)
        return input, vq_loss
            
class YOLODecoder(torch.nn.Module):
    
    def __init__(self, *args, **kwargs):
        super(YOLODecoder, self).__init__(*args, **kwargs)
        
        r = 2 ; w = 2
        c_in = [int(128*r), 128, 64] 
        c_out = [128, 64, 64] 
        self.layers = []
        for c1, c2 in zip(c_in, c_out):
            self.layers.append(YOLODecoderBlock(c1*w, c2*w)) 
        self.layers = torch.nn.ModuleList(self.layers)
        """
        self.pad = lambda x: torch.nn.functional.pad(x, 12)
        self.crop = lambda x: crop_image(x, 12)
        self.up = torch.nn.Upsample(scale_factor=2, mode="bilinear")
        self.c2f = C2fTransposed(64, 64, n=2, shortcut=True)
        self.last_layer = torch.nn.Sequential(
                               torch.nn.ConvTranspose2d(32, 32, 3, bias=True, stride=2), # 320 
                               torch.nn.SiLU(),
                               BatchNorm2d(32),
                               torch.nn.ConvTranspose2d(32, 8, 3, bias=True, stride=2), # 320
                               torch.nn.Conv2d(8, 3, 3, bias=True), # 320
                          ) 
        """
        """
        self.last_layer = torch.nn.Sequential(
                               torch.nn.Upsample(scale_factor=2, mode="bilinear"), # 320 
                               C2fTransposed(64*w, 64, n=2, shortcut=True),
                               torch.nn.Upsample(scale_factor=2, mode="bilinear"), # 640 
                               C2fTransposed(64, 32, n=2, shortcut=True),
                               torch.nn.ConvTranspose2d(32, 8, 3, bias=True), 
                               torch.nn.Conv2d(8, 3, 3, bias=True), # 320
                          )      
        """   
        self.last_layer = torch.nn.Sequential( 
                               torch.nn.Upsample(scale_factor=2, mode="nearest"),
                               torch.nn.ConvTranspose2d(64*w, 64, 3, padding=1, bias=False), 
                               torch.nn.SiLU(),
                               BatchNorm2d(64), 
                               torch.nn.ConvTranspose2d(64, 64, 3, padding=1, bias=False), 
                               torch.nn.SiLU(), 
                               BatchNorm2d(64), 
                               torch.nn.Upsample(scale_factor=2, mode="nearest"), 
                               torch.nn.ConvTranspose2d(64, 32, 3, padding=1, bias=False), 
                               torch.nn.SiLU(), 
                               BatchNorm2d(32), 
                               torch.nn.ConvTranspose2d(32, 32, 3, padding=1, bias=False), 
                               torch.nn.SiLU(), 
                               BatchNorm2d(32),  
                               torch.nn.ConvTranspose2d(32, 8, 3, bias=True), 
                               torch.nn.Conv2d(8, 3, 3, bias=True), # 320
                          ) 
        self.hardtanh = torch.nn.Hardtanh(-1, 1)
        # self.ood_classifier = resnet_err()
        
    def forward(self, input, warmup=1):
        skips = input[::-1] ; input = input[-1] 
        vq_losses = [] 
        for layer, skip in zip(self.layers, skips): 
            input = (input+skip)/2 
            input, vq_loss = layer(input, warmup=warmup)
            vq_losses.append(vq_loss)
        """
        input = self.up(input)
        input = self.pad(input)
        input = self.c2f(input)
        input = self.crop(input)
        """
        input = self.last_layer(input)
        # ood classif with vq_losses
        vq_losses = [torch.mean(_) for _ in vq_losses]
        input = self.hardtanh(input)
        input = (input + 1)/2
        return input, vq_losses
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        