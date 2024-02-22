import numpy as np
import glob, os, sys
import re
import math
import random 
from joblib import Parallel, delayed, parallel_backend
import math

# os.environ['WANDB_MODE']="disabled" 
import copy
import yaml
import wandb
import numpy as np
import shutil
from shutil import copyfile
import hydra
import uuid
from pathlib import Path
from omegaconf import DictConfig
from omegaconf.listconfig import ListConfig

# Make a directory to copy the best weights file to
# if not os.path.exists(DIR+'/output'):
#     os.mkdir(DIR+'/output')

# from data_process_opt import data_process_save
from torch import nn, optim
import pandas as pd
import subprocess
import PIL
import gc

from joblib.externals.loky.backend.context import get_context
from torch.utils.data import DataLoader
import torch

sys.path.append(os.path.join(os.getcwd(), "ultralytics")) 
from ultralytics import YOLO

from ultralytics.utils.torch_utils import (EarlyStopping, ModelEMA, de_parallel, init_seeds, one_cycle, select_device,
                                           strip_optimizer)
                                           
from joblib.externals.loky.backend.context import get_context
multiprocessing_context = get_context('loky') 

import multiprocessing
torch.multiprocessing.set_start_method('spawn', force=True)

ddd = "/auto/home/users/d/a/darimez/DOTA" 
        
class FedYOLO(nn.Module):
    
    def __init__(self, 
                 init_model = "", 
                 data_yaml = "", 
                 data_path = "", 
                 data_sizes = "", 
                 val=1000,
                 test=2000,
                 labelled_perc=0.1,
                 name = "", 
                 n_models = 5, 
                 annotation_budget = 0.05,
                 train_args = {"all":{}}, 
                 validation_args = {}, 
                 compute_norm=True,
                 active_strategy="random",
                 n_jobs=0, 
                 aggregate_fn="fedavg",
                 keep_backgrouds=True,
                 *args, **kwargs):
        super(FedYOLO, self).__init__(*args, **kwargs)
        self.init_model = init_model
        self.n_models = n_models
        self.n_jobs = n_jobs if n_jobs>=0 else n_models
        self.aggregate_fn = aggregate_fn
        kd = aggregate_fn=="kd"
        self.global_model = YOLO(init_model) 
        #self.models = Parallel(n_jobs=self.n_jobs, timeout=6000, pre_dispatch="2*n_jobs", verbose=100)(
        #                      delayed(copy.deepcopy)(self.global_model) for _ in range(n_models) )  
        self.models = [copy.deepcopy(self.global_model) for _ in range(n_models)] 
        self.train_args = train_args
        self.validation_args = validation_args
        self.compute_norm = compute_norm
        self.device = torch.device("cuda")
        self.active_strategy = active_strategy
        self.annotation_budget = annotation_budget 
        self.name = name
        self.run_id = [None for _ in range(self.n_models)]
        self.class_counts = [[0 for _ in range(15)] for _ in range(n_models)] 
        self.max_batch = [None for _ in range(self.n_models)]
        self.trainloaders = [None for _ in range(self.n_models)]
        
        yaml_path = "/".join(data_yaml.split("/")[:-1])
        self.data_yamls = [yaml_path + "/" + str(i) for i in range(n_models)]
        [os.makedirs(p) for p in self.data_yamls if not os.path.isdir(p)]  
        os.makedirs(yaml_path + "/42")
        self.data_yamls = [yaml_path + "/" + str(i) + "/" + "data.yaml" for i in range(n_models)]
        
        for id, data_yaml in enumerate(self.data_yamls):
            self.train_args[f"{id}"] = {}
            self.train_args[f"{id}"]["data"] = data_yaml
        
        self.train_args["42"] = {"data": yaml_path + "/42/" + "data.yaml"} 
            
        self.data_path = data_path
        create_mixte_dataset(images_dir=data_path, 
                             txt_dir=str(yaml_path),
                             data_sizes=data_sizes ,
                             labelled_perc=labelled_perc,
                             val=val,
                             test=test,
                             kd=kd, keep_backgrouds=keep_backgrouds)
    """    
    def get_run(self, id=42):
        run_id = 0 if id==42 else id + 1
        global_args = {**self.train_args["0"], **self.train_args["all"]}
        global_args["epochs"] = 1
        global_args["name"] = self.name + f"_{id}" 
        wandb_config = self.global_model.trainer.wandb_config  
        wandb_config = {**wandb_config, **vars(self.global_model.trainer.args)}
        run = wandb.init(project=self.global_model.trainer.args.project or 'YOLOv8', name=self.global_model.trainer.args.name, 
                      config=wandb_config, entity=self.global_model.trainer.entity, group=self.global_model.trainer.group,
                      resume=True, id=self.run_id[run_id]) 
        return run
    """
    def get_run(self, id=42):
        """
        run_id = 0 if id==42 else id + 1
        global_args = {**self.train_args["0"], **self.train_args["all"]}
        global_args["epochs"] = 1
        global_args["name"] = self.name #  + f"_{id}" 
        wandb_config = self.global_model.trainer.wandb_config  
        wandb_config = {**wandb_config, **vars(self.global_model.trainer.args)}
        run = wandb.init(project=self.global_model.trainer.args.project or 'YOLOv8', name=self.global_model.trainer.args.name, 
                      config=wandb_config, entity=self.global_model.trainer.entity, group=self.global_model.trainer.group,
                      resume=True, id=self.run_id[run_id]) 
        """
        return self.global_model.trainer.run
        
    def get_param_names(self, model):
        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                yield (module_name, param_name)
        
    def KD(self, step, save_path, a, f):
        compute_norm=self.compute_norm
        if self.compute_norm: 
            update_Norm = 0
        # Nd = torch.Tensor([model.trainer.data_size for model in self.models])
        Nd = torch.Tensor([1 for model in self.models])  
        # Nd_head = torch.Tensor([len(self.models)*torch.Tensor(clc)/sum(clc) for clc in zip(self.class_counts)]) # nc x 3
        ntot = sum(Nd) ; Nd = Nd/ntot ; Nd = Nd.to(self.device)  
        # create_dataset_kd(self.models, Nd, self.train_args["42"]["data"])
        new_model = YOLO(self.init_model) # .trainer.model.to(self.device).float()) # model.trainer. 
        kd_args = {**self.train_args["all"]}
        kd_args["epochs"] = (f+1)*self.train_args["all"]["epochs"] # train the global model as many epochs the last global model + local epochs
        new_model.train(kd=True, model=self.init_weight, 
                        models=self.models, Nd=Nd, 
                        start_epoch_log=0, 
                        run=self.get_run(),
                        name= self.name + f"_{a}_{f}_42",
                        **self.train_args[f"42"],
                        **kd_args) 
        with torch.no_grad(): 
            for new_param, params in zip(self.init_weight.to(self.device).float().parameters(), new_model.to(self.device).float().parameters()):
                update_Norm += torch.sum((new_param - params)**2) 
                
            self.global_model.model = new_model.trainer.model.to(self.device).float()
            self.global_model.trainer.model = new_model.trainer.model.to(self.device).float() 
            ckpt = {'epoch': 0,
                    'best_fitness': 0,
                    'model': new_model.trainer.model.to(self.device).float(),
                    'ema': None,
                    'updates': None,
                    'optimizer': None,
                    'train_args': vars(self.global_model.trainer.args),  # save as dict
                    'train_metrics': None,
                    'train_results': None,
                    'date': None,
                    'version': None} 
            torch.save(ckpt, save_path) ; new_model = None 
            
            if self.compute_norm:
                try: 
                    run = self.get_run()
                    run.log({**{f"Weigth {nnn}":nd for nnn,nd in enumerate(Nd)}, "Update norm": update_Norm**0.5, "step":int(step)} ) # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    # run.finish()
                except Exception as err:
                    print(err)
                    
    def fedavg(self, step, save_path, a, f):
        compute_norm=self.compute_norm
        with torch.no_grad(): 
            if self.compute_norm:
                total_norm = torch.zeros(self.n_models).to(self.device)
                update_Norm = 0
            # Nd = torch.Tensor([model.trainer.data_size for model in self.models])
            Nd = torch.Tensor([1 for model in self.models])  
            # Nd_head = torch.Tensor([len(self.models)*torch.Tensor(clc)/sum(clc) for clc in zip(self.class_counts)]) # nc x 3
            ntot = sum(Nd) ; Nd = Nd/ntot ; Nd = Nd.to(self.device) 
            new_model = copy.deepcopy(self.global_model.trainer.model.to(self.device).float()) # model.trainer.
            names = self.get_param_names(new_model)
            for new_param, *params in zip(new_model.parameters(), *[model.to(self.device).float().parameters() for model in self.models]):
                name = next(names)
                # Nd = Nd_bckb
                if "22" in name and False:
                    Nd = torch.transpose(Nd_head, (0,1))
                # print(param.data.shape)
                updates = torch.stack([param.data.to(self.device)*nd for param, nd in zip(params, Nd)])  
                # compute updates' norm
                if self.compute_norm:
                    try:
                        new = torch.stack([new_param.data.to(self.device)*nd for nd in Nd]) 
                        new = torch.sum((new - updates)**2, dim=tuple(np.arange(1,len(new.shape)).astype(int)))
                        total_norm += new
                    except Exception as err:
                        print(err)
                # compute weighted average  
                # print(updates.min(), updates.max())
                mean = torch.sum(updates, dim=0)
                # print(mean.min(), mean.max())
                update_Norm += torch.sum((new_param - mean)**2)
                new_param.data.copy_(mean)
                
            self.global_model.model = new_model 
            self.global_model.trainer.model = new_model  
            ckpt = {'epoch': 0,
                    'best_fitness': 0,
                    'model': new_model,
                    'ema': None,
                    'updates': None,
                    'optimizer': None,
                    'train_args': vars(self.global_model.trainer.args),  # save as dict
                    'train_metrics': None,
                    'train_results': None,
                    'date': None,
                    'version': None} 
            torch.save(ckpt, save_path) ; new_model = None 
            
            if self.compute_norm:
                try:
                    total_norm = total_norm ** 0.5 
                    run = self.get_run()
                    run.log({**{f"Gradient norm {nnn}":mmm/Nd[nnn] for nnn,mmm in enumerate(total_norm)}, 
                             **{f"Weigth {nnn}":nd for nnn,nd in enumerate(Nd)}, "Update norm": update_Norm**0.5, "step":int(step)} ) # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    # run.finish()
                except Exception as err:
                    print(err)
                
        
    def aggregate(self, step, save_path, a, f):
        aggregate_fn=self.aggregate_fn
        if aggregate_fn=="fedavg":
            return self.fedavg(step, save_path, a, f) 
        elif aggregate_fn=="kd":
            return self.KD(step, save_path, a, f) 
        raise "Unimplemented aggregation function, fell free to add it"
        
    def active_query(self):
        
        if self.active_strategy=="random":
            query = query_random
        elif self.active_strategy=="confidence":
            query = query_confidence
        elif self.active_strategy=="coreset":
            query = query_coreset
        else:
            raise "Unimplemented active query strategy, fell free to add it"
             
        if self.n_jobs>0:  
            # Parallelize the training of individual models in the ensemble 
            torch.multiprocessing.set_start_method(torch.multiprocessing.get_start_method(), force=True) 
            processes = []
            for model in self.models:
                process = torch.multiprocessing.Process(target=query, args=(model, data_yaml, self.annotation_budget))
                processes.append(process)
                process.start()
            # Wait for all processes to finish
            for process in processes:
                process.join()
            """
            with multiprocessing.context.DefaultContext(multiprocessing_context) as context:
                # context._actual_context = multiprocessing_context
                # with parallel_backend('loky', n_jobs=self.n_jobs):
                Parallel(n_jobs=self.n_jobs, timeout=6000, pre_dispatch="n_jobs", verbose=100)(
                                                          delayed(query)(model, data_yaml, self.annotation_budget) 
                                                          for model, data_yaml in zip(self.models, self.data_yamls) ) 
            """ 
        else:
            for model, data_yaml in zip(self.models, self.data_yamls):
                query(model, data_yaml, self.annotation_budget) 
            
    def _train(self, im, model, setup=True, round=0, resume=False, alround=0, run=None):  
        
        model.train(name=self.name + f"_42" if im==-1 else self.name + f"_{alround}_{round}_{im}",
                    **self.train_args[f"{im}"],
                    **self.train_args["all"], 
                    start_epoch_log=self.train_args["all"]["epochs"] if round>0 else 0, 
                    trainer=model.trainer, resume=resume, run=run, model=model.model) 
        # model.trainer.run.finish()
        nb = len(model.trainer.train_loader)  # number of batches
        nw = max(int(model.trainer.args.warmup_epochs * nb), 100)
        ni = self.train_args["all"]["epochs"]*nb
        model.trainer.args.warmup_epochs = max(0, model.trainer.args.warmup_epochs - self.train_args["all"]["epochs"]) 
        # model.trainer.args.warmup_epochs = 5
        if False:
            model.trainer.args.warmup_epochs = model.trainer.args.warmup_epochs + model.trainer.start_epoch 
        # print("WARMUP :", model.trainer.args.warmup_epochs, model.trainer.start_epoch)
        # torch.save({"model":model, "a":a, "f":f}, save_path) 
        return model
        
    def train2(self, im, model, alround, round):
        TRAIN = True
        save_path = "/".join(*self.train_args[f"{im}"]["data"].split("/")[:-1], "model.pt")
        if os.path.exists(save_path): # checks if this model have a ckpt already
            ckpt = torch.load(save_path)
            al, fed = ckpt["a"], ckpt["f"]
            if al*1000 + fed <= alround*1000 + round: # ckpt does not corresponds to end of last fed round -> train  
                self.models[im] = ckpt["model"] 
                TRAIN = False
            else:
                model = ckpt["model"]  
        if round==0: # needs to reload dataset
            # model._setup_train()
            self.run_id[im] = model.trainer.run.id 
            setup = True
        else: # do not re-create dataset  
            self._setup_train(model)  
        if TRAIN:
            run =  self.get_run()
            model.trainer.run = run 
            setup = False # setup already done
            self.models[im] = self._train(im, model, setup=setup, round=round) 

    def setup_train(self):
        print("Setup")
        global_args = {**copy.deepcopy(self.train_args["0"]), **copy.deepcopy(self.train_args["all"])}
        global_args["epochs"] = 1
        global_args["name"] = self.name + "_00_42"
        global_args["freeze"] = 42 # higher than the number of layers for safety
        print(global_args["name"])  
        self.global_model.train(**global_args, setup_log=True) 
        self.run_id[0] = self.run_id[0] or self.global_model.trainer.run.id 
        # self.global_model.trainer.run.finish()
       
    def _setup_train(self, model):
    
        batch_size = model.trainer.batch_size #  // max(world_size, 1)
        model.trainer.train_loader = model.trainer.train_loader # get_dataloader(model.trainer.trainset, batch_size=batch_size, rank=-1, mode='train')
        if True: 
            model.trainer.ema = ModelEMA(model.trainer.model)
            if model.trainer.args.plots:
                model.trainer.plot_training_labels()

        # Optimizer
        model.trainer.accumulate = max(round(model.trainer.args.nbs / model.trainer.batch_size), 1)  # accumulate loss before optimizing
        weight_decay = model.trainer.args.weight_decay * model.trainer.batch_size * model.trainer.accumulate / model.trainer.args.nbs  # scale weight_decay
        iterations = math.ceil(len(model.trainer.train_loader.dataset) / max(model.trainer.batch_size, model.trainer.args.nbs)) * model.trainer.epochs
        model.trainer.optimizer = model.trainer.build_optimizer(model=model.trainer.model,
                                                                name=model.trainer.args.optimizer,
                                                                lr=model.trainer.args.lr0,
                                                                momentum=model.trainer.args.momentum,
                                                                decay=weight_decay,
                                                                iterations=iterations)
        # Scheduler
        if model.trainer.args.cos_lr:
            model.trainer.lf = one_cycle(1, model.trainer.args.lrf, model.trainer.epochs)  # cosine 1->hyp['lrf']
        else:
            model.trainer.lf = lambda x: (1 - x / model.trainer.epochs) * (1.0 - model.trainer.args.lrf) + model.trainer.args.lrf  # linear
        model.trainer.scheduler = optim.lr_scheduler.LambdaLR(model.trainer.optimizer, lr_lambda=model.trainer.lf)
        model.trainer.stopper, model.trainer.stop = EarlyStopping(patience=model.trainer.args.patience), False    


    def train(self, al_rounds=10):  
        
        start_al, start_fed = 0, 0
        save_path = "/".join((*self.train_args["0"]["data"].split("/")[:-2], "global_model.pt"))
        if os.path.exists(save_path): # start from checkpoint
            ckpt = torch.load(save_path)
            start_al, start_fed = ckpt["a"], ckpt["f"]
            self.global_model = ckpt["model"]
        else:
            self.setup_train()
            torch.save({"model":self.global_model.trainer.model, "a":0, "f":0}, save_path)
            torch.save({"model":self.global_model.trainer.model, "a":0, "f":0}, save_path.replace("global_model","init_model"))
        global_save_path = save_path
        init_save_path = save_path.replace("global_model","init_model")
        self.init_weight = copy.copy(self.global_model.trainer.model)
        
        fed_rounds = 300//self.train_args["all"]["epochs"]
        for alround in range(start_al, al_rounds):   
            if alround>0:
                self.validate(alround, -1)  
                # torch.save({"model":self.global_model, "a":alround, "f":0}, global_save_path) 
            print(f"Active Learning round {alround}")
            # Query new data following AL strategy 
            if alround>0:
                self.active_query() 
            if alround==2:
                self.aggregate_fn = "kd"
            for round in range(start_fed, fed_rounds):  
                print(f"Federated aggregation {round}")
                gc.collect()
                torch.cuda.empty_cache()
                # self.aggregate(alround*fed_rounds + round)
                # Train local models
                if self.n_jobs>0:
                    # Parallelize the training of individual models in the ensemble
                    torch.multiprocessing.set_start_method(torch.multiprocessing.get_start_method(), force=True)
                    # setups = [torch.multiprocessing.Process(target=model.trainer._setup_train, args=()) for model in self.models]
                    processes = [torch.multiprocessing.Process(target=self.train2, args=(im, model, alround, round)) for im, model in enumerate(self.models)]
                    """
                    for i in range(len(self.models)):
                        if i>0:
                            setups[i-1].join() # wait for end of last setup before beginning next one
                            processes[i-1].start()
                        setups[i].start() 
                    setups[-1].join()
                    processes[-1].start() 
                    """
                    """
                    for im in enumerate(self.models): 
                        process = torch.multiprocessing.Process(target=self._train, args=(im, model)) 
                        processes.append(process) 
                        process.start() 
                    """ 
                    for process in processes:
                        process.start()
                    # Wait for all processes to finish
                    for process in processes:
                        process.join()
                    """
                    with multiprocessing.context.DefaultContext(multiprocessing_context) as context:
                        # context._actual_context = multiprocessing_context
                        # with parallel_backend('loky', n_jobs=self.n_jobs):
                        self.models = Parallel(n_jobs=self.n_jobs, timeout=6000, pre_dispatch="n_jobs", verbose=100)(
                                              delayed(_train)(im, model) for im, model in enumerate(self.models) )  
                    """
                else:
                    for im, model in enumerate(self.models):
                        save_path = global_save_path # "/".join((*self.train_args[f"{im}"]["data"].split("/")[:-1], "model.pt"))
                        if round==0:
                            save_path = init_save_path
                        if os.path.exists(save_path):
                            mod = torch.load(save_path)['model']
                        else:
                            print("aie aie aie")
                            # mod.model.load_state_dict(copy.deepcopy(self.init_weight.state_dict()))
                        if round==0:
                            warmup_epochs = 4
                        else:
                            warmup_epochs = 4 + self.train_args["all"]["epochs"]*round
                        new_model = YOLO(mod) 
                        new_model.train(name=self.name + f"_42" if im==-1 else self.name + f"_{alround}_{round}_{im}",
                                        **self.train_args[f"{im}"],
                                        **self.train_args["all"], 
                                        start_epoch_log=self.train_args["all"]["epochs"]*round, 
                                        run=self.get_run(), warmup_epochs=warmup_epochs, model=mod, 
                                        setup=False, max_batch=self.max_batch[im])   
                        if round==0:
                            self.max_batch[im] = len(new_model.trainer.train_loader)*self.train_args["all"]["epochs"]
                            self.trainloaders[im] = [new_model.trainer.train_loader, new_model.trainer.model]
                        else:
                            self.trainloaders[im][1] = new_model.trainer.model
                        self.models[im] = copy.copy(new_model.trainer.model)
                        # self.class_counts[im] = new_model.trainer.class_counts
                        new_model = None
                        """
                        if os.path.exists(save_path): # checks if this model have a ckpt already
                            ckpt = torch.load(save_path)
                            al, fed = ckpt["a"], ckpt["f"]
                            if al*1000 + fed <= alround*1000 + round: # ckpt does not corresponds to end of last fed round -> train  
                                self.models[im] = ckpt["model"]
                                continue # skip training
                            else:
                                model = ckpt["model"] 
                         
                        if round==0: # needs to reload dataset
                            # model._setup_train()
                            # self.run_id[im] = model.trainer.run.id 
                            setup = True
                        else: # do not re-create dataset 
                            ###
                            if self.run_id[im]==None: # create new wandb instance / normally it should not be used
                                model.trainer.run_callbacks('on_pretrain_routine_start')
                                self.run_id[im] = model.trainer.run.id 
                            else: # get old wandb instance back
                                run =  self.get_run(im)
                                model.trainer.run = run
                            ###
                            self._setup_train(model) 
                        run =  self.get_run() 
                        setup = False # setup already done
                        self.models[im] = self._train(im, model, setup=setup, resume=round>0, round=round, alround=alround, run=run) 
                        """
                # Evaluates local models on check set and discard suspicious local models
                # self.byzantine_agreement()
                # aggregate the updates and update global model
                self.aggregate((alround*fed_rounds + round)*self.train_args["all"]["epochs"], global_save_path, alround, fed_rounds)
                # evaluate global model on validation data
                # self.validate(alround, round)
                if round>0: 
                    pass 
                    # torch.save({"model":self.global_model, "a":alround, "f":round + 1}, global_save_path)
                # update local models with global model's weights 
                """
                for im, model in enumerate(self.models): 
                    model.trainer.model.load_state_dict(self.global_model.model.state_dict()) 
                    model.model.load_state_dict(self.global_model.model.state_dict()) 
                    # self.run_id[im] = None
                """
        self.global_model.trainer.run.finish()
        
    def validate(self, alround, fedround):
        self.global_model.trainer.args.name = self.name + f"_{alround}_42"
        self.global_model.trainer.epoch = (fedround + 1)*self.train_args["all"]["epochs"]
        self.global_model.trainer.model = self.global_model.model
        # self.global_model.trainer.run = self.get_run()
        self.global_model.trainer.metrics, self.global_model.trainer.fitness = self.global_model.trainer.validate()
        self.global_model.trainer.save_metrics(metrics={**self.global_model.trainer.label_loss_items(self.global_model.trainer.tloss), 
                                                        **self.global_model.trainer.metrics, 
                                                        **self.global_model.trainer.lr})  
        # Save model
        self.global_model.trainer.save_model()
        self.global_model.trainer.run_callbacks('on_model_save') 
        self.global_model.trainer.run_callbacks('on_fit_epoch_end') 
        # self.global_model.trainer.run.finish()
        # self.global_model.validate(**self.validation_args)

"""
AL sampling
"""        

class Coreset_Greedy:
    def __init__(self, all_pts):
        self.all_pts = np.array(all_pts)
        self.dset_size = len(all_pts)
        self.min_distances = None
        self.already_selected = [] 
        
    def update_dist(self, centers, only_new=True, reset_dist=False):
        if reset_dist:
            self.min_distances = None
        if only_new:
            centers = [p for p in centers if p not in self.already_selected]

        if centers is not None: #is not None:
            x = self.all_pts[centers]  # pick only centers

            dist = pairwise_distances(self.all_pts, x, metric='euclidean')
            if self.min_distances is None:
                self.min_distances = np.min(dist, axis=1).reshape(-1, 1)
            else:
                self.min_distances = np.minimum(self.min_distances, dist)

    def sample(self, already_selected, sample_size):
        # initially updating the distances
        if not self.already_selected == []:
            self.update_dist(already_selected, only_new=False, reset_dist=True)
        self.already_selected = already_selected
        new_batch = []
        for i in range(sample_size):
            print("coreset i", i)
            if self.already_selected == []:
                ind = np.random.choice(np.arange(self.dset_size))
            else:
                ind = np.argmax(self.min_distances)
            already_selected.append(ind)
            #assert ind not in already_selected
            self.update_dist([ind], only_new=False, reset_dist=False)
            new_batch.append(ind)

        return new_batch, max(self.min_distances)

def query_coreset(model, real_data, gen_data, data_yaml, sel, selected_path, fold, it):
    print("START coreset")
    used_data = None
    with open(data_yaml, 'r') as _file:
        used_data = yaml.safe_load(_file) 

    old_train = used_data["train"] + ""
    train = used_data["train"]
    if not fold in train:
        train = used_data["train"].replace("train", f"/{fold}/train_" + str(gen_data[0]).split("/")[-2])

    data_yaml_file = used_data
    data_yaml_file["train"] = train
    
    with open(data_yaml, 'w') as _file:
         yaml.dump(used_data, _file) 

    with open(old_train, 'r') as _file:
        used_data = _file.readlines()

    U = [u.replace('\n','') for u in used_data]
    used_data = [u.split("/")[-1] for u in U] 
    real_data = [r for r in real_data if not str(r).split("/")[-1] in used_data]
    gen_data = [g for g in gen_data if not str(g).split("/")[-1] in used_data]

    model.model.query = True 
    real_features = [] 
    gen_features = []
    for i, p in enumerate(model.predict(source=gen_data)):
        gen_features += [p.cpu().numpy()] 

    # generated images selection with AL 
    coreset = Coreset_Greedy(gen_features)
    gen_idx_selected, max_distance = coreset.sample(used_data, sel)  
    synt_dataset_selected = np.array(gen_data)[gen_idx_selected] 
    selected = [str(selected_path / s) for s in synt_dataset_selected]
    # print("selected", selected); 
    used_data = list(U) + selected
    
    if it==1 and False:
        data_yaml_file = used_data_
        train = used_data_["train"].replace("train", f"{fold.split('/')[-1]}/train_" + str(gen_data[0]).split("/")[-2])
        data_yaml_file["train"] = train 
        with open(data_yaml, 'w') as _file: 
           yaml.dump(data_yaml_file, _file)
    with open(train, "w") as _file:
        _file.write("\n".join(used_data))
    print("STOP coreset")

def query_confidence(model, real_data, gen_data, data_yaml, sel, selected_path, fold, it):
    print("START query")

    used_data = None
    with open(data_yaml, 'r') as _file:
        used_data = yaml.safe_load(_file) 

    old_train = used_data["train"] + ""
    train = used_data["train"]
    if not fold in train:
        train = used_data["train"].replace("train", f"/{fold}/train_" + str(gen_data[0]).split("/")[-2])
        
    data_yaml_file = used_data
    data_yaml_file["train"] = train
    
    with open(data_yaml, 'w') as _file:
         yaml.dump(used_data, _file) 

    with open(old_train, 'r') as _file: # train plutot que old_train
        used_data = _file.readlines()

    U = [u.replace('\n','') for u in used_data]
    used_data = [u.split("/")[-1] for u in U]
    gen_data = [g for g in gen_data if not str(g).split("/")[-1] in used_data]

    results_gen = model.predict(source = gen_data)
    results_gen = [r.boxes.conf.cpu().numpy() for r in results_gen]
    results_gen = [r.max() if len(r)>=1 else 0.9 for r in results_gen]
    results_gen = np.array(gen_data)[np.argsort(results_gen)]

    model.model.query = True  
    gen_features = []
    for i, p in enumerate(model.predict(source=gen_data)):
         gen_features += [p.cpu().numpy()]
    # generated images selection with AL 
    coreset = Coreset_Greedy(gen_features)
    gen_idx_selected, max_distance = coreset.sample([], sel)
    synt_dataset_selected = np.array(gen_data)[gen_idx_selected]

    selected = [str(selected_path / s) for s in synt_dataset_selected]
    used_data = list(U) + selected
    if it==1 and False:
        data_yaml_file = used_data_
        train = used_data_["train"].replace("train", f"{fold.split('/')[-1]}/train_" + str(gen_data[0]).split("/")[-2])
        data_yaml_file["train"] = train; # print("train it==1", train)
        with open(data_yaml, 'w') as _file:
            yaml.dump(data_yaml_file, _file)
    with open(train, "w") as _file: #train
        _file.write("\n".join(used_data))
    print("STOP query")
    
def query_random(model, data_yaml, sel):
    print("START query") 
    used_data = None
    with open(data_yaml, 'r') as _file:
        used_data = yaml.safe_load(_file) 

    unlabelled = used_data["unlabelled"] + ""
    train = used_data["train"] 
        
    data_yaml_file = used_data 
    with open(data_yaml, 'w') as _file:
         yaml.dump(used_data, _file)  
    with open(train, 'r') as _file:
        used_data = _file.readlines()

    data_path = '/'.join(used_data[-1].replace('\n','').split('/')[:-1])

    U = [u.replace('\n','') for u in used_data]
    used_data = [u.split("/")[-1] for u in U]   
     
    with open(unlabelled, 'r') as _file:
        real_data = _file.readlines()
    real_data = [r.replace('\n','') for r in real_data]
    real_data = [r.split("/")[-1] for r in real_data]
    
    sel = int(sel*len(real_data))
    real_data = [r for r in real_data if not str(r) in used_data]
    
    selected = np.array(real_data)[:sel] 
    selected = [ data_path + "/" + s for s in selected if not str(s) in used_data]
    used_data = list(U) + list(selected) 
    with open(train, "w") as _file:  
        _file.write("\n".join(used_data))
    print("STOP query")
     
      
"""
Dataset 
"""
def create_dataset_kd(model, weights, data_yaml):
    
    with open(data_yaml, "r") as file:
        data = yaml.safe_load(file)
        
    with open(data["train"], "r") as file:
        train = file.readlines()

    for img in train:
        img = train + "/" + img
        with open(train[0].replace("images","labels").replace(".png",".txt"),"w") as file:  
            for result in non_max_suppression(model(img)):
                boxes = result.boxes.cpu().numpy()
                for box in boxes:
                    cls = int(box.cls[0])  
                    conf = int(box.conf[0]*100)
                    bx = box.xywh.tolist()
                    out = f"{cls} {bx[0]} {bx[1]} {bx[2]} {bx[3]}\n"
                    file.write(out)  


def create_mixte_dataset(images_dir = "", 
                         txt_dir = "",
                         data_sizes = [1000, 1000, 1000, 1000, 1000],
                         labelled_perc = 0.1,
                         val = 1000,
                         test = 2000,
                         formats = ['jpg', 'png', 'jpeg'], kd=False, keep_backgrouds=True):
    """
    Construct the txt file containing a percentage of real and synthetic data

    :param str real_images_dir: path to the folder containing real images
    :param str synth_images_dir: path to the folder containing synthetic images
    :param str txt_dir: path used to create the txt file
    :param float per_synth_data: percentage of synthetic data compared to real (ranges in [0, 1])

    :return: None
    :rtype: NoneType
    """
    txt_dir = Path(txt_dir)
    classes = [ 'plane', 'ship', 'storage-tank', 'baseball-diamond', 'tennis-court', 
                'basketball-court', 'ground-track-field', 'harbor', 'bridge', 'large-vehicle', 
                'small-vehicle', 'helicopter', 'roundabout', 'soccer-ball-field', 'swimming-pool' ]
    # classes = ["person"]
    classes = ["vehicle"]
    classes_to_filter = np.ones(len(classes))*1000000 
    if isinstance(images_dir,ListConfig):
        images_dirs = images_dir
        for id, (data_size, images_dir) in enumerate(zip(data_sizes,images_dirs)): 
            images = list_images(images_dir, formats, val + test + data_size, None, keep_backgrouds=keep_backgrouds)
            train_images = sorted(images)
        
            # shuffle images
            random.Random(42).shuffle(train_images) 
         
            train_txt_path = txt_dir / str(id) / 'train.txt'
            unlab_txt_path = txt_dir / str(id) / 'unlabelled.txt'
            val_txt_path = txt_dir / str(id) / 'val.txt'
            test_txt_path = txt_dir / str(id) / 'test.txt'
            data_yaml_path = txt_dir / str(id) / 'data.yaml'
            
            this_train_images = train_images[:data_size]
            labelled = int(len(this_train_images)*labelled_perc)
            labelled, unlabelled = ( this_train_images[:labelled], this_train_images ) 
            
            if isinstance(images_dir,ListConfig):
                images_dir = [_.replace(f"{os.sep}train",f"{os.sep}test") for _ in images_dir]
            images = list_images(images_dir, formats, val + test + data_size, None, keep_backgrouds=keep_backgrouds) 
            val_images = images
            test_images = []
            
            with open(train_txt_path, 'w') as f:
                f.write('\n'.join(labelled)) 
            with open(unlab_txt_path, 'w') as f:
                f.write('\n'.join(unlabelled)) 
            with open(val_txt_path, 'w') as f:
                f.write('\n'.join(val_images)) 
            with open(test_txt_path, 'w') as f:
                f.write('\n'.join(test_images))
    
            create_yaml_file(data_yaml_path, train_txt_path, unlab_txt_path, val_txt_path, test_txt_path, classes)
    else: 
        images = list_images(images_dir, formats, val + test + sum(data_sizes), None, keep_backgrouds=keep_backgrouds)
        val_images = images[:val]
        test_images = images[val:val+test]
        train_images = sorted(images[val+test:])
    
        # shuffle images
        random.Random(42).shuffle(train_images) 
    
        for id, data_size in enumerate(data_sizes):
        
            train_txt_path = txt_dir / str(id) / 'train.txt'
            unlab_txt_path = txt_dir / str(id) / 'unlabelled.txt'
            val_txt_path = txt_dir / str(id) / 'val.txt'
            test_txt_path = txt_dir / str(id) / 'test.txt'
            data_yaml_path = txt_dir / str(id) / 'data.yaml'
        
            already_selected = sum(data_sizes[:id+1])
            this_train_images = train_images[already_selected-data_size:already_selected]
            labelled = int(len(this_train_images)*labelled_perc)
            labelled, unlabelled = ( this_train_images[:labelled], this_train_images ) 
            
            with open(train_txt_path, 'w') as f:
                f.write('\n'.join(labelled)) 
            with open(unlab_txt_path, 'w') as f:
                f.write('\n'.join(unlabelled)) 
            with open(val_txt_path, 'w') as f:
                f.write('\n'.join(val_images)) 
            with open(test_txt_path, 'w') as f:
                f.write('\n'.join(test_images))
    
            create_yaml_file(data_yaml_path, train_txt_path, unlab_txt_path, val_txt_path, test_txt_path, classes)

    if kd:
        id, data_size = 42, 1000
    
        train_txt_path = txt_dir / str(id) / 'train.txt'
        unlab_txt_path = txt_dir / str(id) / 'unlabelled.txt'
        val_txt_path = txt_dir / str(id) / 'val.txt'
        test_txt_path = txt_dir / str(id) / 'test.txt'
        data_yaml_path = txt_dir / str(id) / 'data.yaml'
    
        already_selected = sum(data_sizes[:id+1])
        this_train_images = train_images[already_selected-data_size:already_selected]
        labelled = int(len(this_train_images)*labelled_perc)
        labelled, unlabelled = ( this_train_images[:labelled], this_train_images ) 
        """
        os.makedirs(txt_dir / str(id) / "images")
        os.makedirs(txt_dir / str(id) / "labels")
        for im in unlabelled:
            os.system(f"cp {im} {txt_dir / str(id) / 'images'}")
        """
        with open(train_txt_path, 'w') as f:
            f.write('\n'.join(labelled))  
        with open(val_txt_path, 'w') as f:
            f.write('\n'.join(val_images)) 
        with open(test_txt_path, 'w') as f:
            f.write('\n'.join(test_images))

        create_yaml_file(data_yaml_path, train_txt_path, unlab_txt_path, val_txt_path, test_txt_path, classes)

def filter_classes(images, classes_to_filter, keep_backgrouds=True, formats=[]):

    if classes_to_filter is None and keep_backgrouds:
        return images
    
    new_images = [] 
    for im in images:
        label = im.replace("images","labels")
        for format in formats:
            label = label.replace(f".{format}",".txt")
        label = np.loadtxt(label)
        if len(label)==0:
            if keep_backgrouds:
                new_images.append(im)
        else:
            if classes_to_filter is None:
                new_images.append(im)
            else:
                new_label = [] ; classes_counts = np.zeros(len(classes_to_filter))
                for lab in label:
                    if classes_to_filter[lab[0]]<0:
                        continue  # skip to control amount of instances
                    classes_counts[lab[0]] += 1 
                classes_to_filter -= classes_counts
                if len(new_label)==0 and keep_backgrouds:
                    new_images.append(im) 
                elif len(new_label)>0:
                    new_images.append(im) 
    return new_images
        

def list_images(images_path: Path, formats=[], max_=None, classes_to_filter=None, keep_backgrouds=True):
    
    images = []
    if isinstance(images_path, ListConfig) or isinstance(images_path, list):
        for images_p in images_path:
            images += list_images(images_p, formats, max_, classes_to_filter, keep_backgrouds)
    else:
        images_path = Path(images_path) / 'images'
        for format in formats:
            images += [
                *glob.glob(str(images_path.absolute()) + f'/*.{format}')
            ]
        images = filter_classes(images, classes_to_filter, keep_backgrouds, formats=formats)
    return images 

def create_yaml_file(save_path: Path, train: Path, unlab: Path, val: Path, test: Path, classes=[]):
    """
    Construct the yaml file

    :param pathlib.Path txt_dir: path used to create the txt files
    :param pathlib.Path yaml_dir: path used to create the yaml file

    :return: None
    :rtype: NoneType
    """
    yaml_file = {
        'train': str(train.absolute()),
        'unlabelled': str(unlab.absolute()),
        'val': str(val.absolute()),
        'test': str(test.absolute()),
        'names': {ii: class_name for ii, class_name in enumerate(classes)}
    }

    with open(save_path, 'w') as file:
        yaml.dump(yaml_file, file)      
        
######################################################################################    
    
# @profile
@hydra.main(version_base=None, config_path=f"..{os.sep}conf", config_name="config")
def main(cfg: DictConfig) -> None:

    base_path = Path(cfg['root'])    
    name = f"{uuid.uuid4().hex.upper()[0:6]}_{cfg['name']}_walt"
    # name = f"{cfg['name']}"
    
    data_yaml_path = base_path / "runs" / name / 'data.yaml'
    # model_path = base_path / "" + ".pt"

    data_yaml_path = str(data_yaml_path.absolute())  
    train_args = {**cfg['optimiser'],
                  **cfg['wandb'], 
                  "group":name }  
    model = FedYOLO( init_model = cfg['init_model'], 
                     data_yaml = data_yaml_path, 
                     name = name, 
                     n_models = cfg["coalition"]['n_models'], 
                     data_path = cfg["data"]["path"], 
                     data_sizes = cfg["data"]["data_sizes"], 
                     val=cfg["data"]["val"],
                     test=cfg["data"]["test"],
                     labelled_perc=cfg["active"]["labelled_perc"],
                     active_strategy=cfg["active"]["strategy"],
                     annotation_budget = cfg["active"]['annotation_budget'],
                     train_args = {"all":train_args}, 
                     validation_args = {}, 
                     compute_norm=cfg["compute_norm"],
                     aggregate_fn="kd",
                     keep_backgrouds=False,
                     ) 
    model.train() # alrounds=cfg["active"]["rounds"])

if __name__ == '__main__':
    main()

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
