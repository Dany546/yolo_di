from my_yolov5.utils.general import xywhn2xyxy
from config import img_shape
import numpy as np
import copy
import torch
from joblib import Parallel, delayed

stdout = None

def demi_gaussian(s):
    x = np.random.normal(1,s)
    while x<0 or x>1:
        x = np.random.normal(1,s)
         
def noo(a, ns, mi, ma):
    x = np.random.normal(a, max(ns/3,0))
    while x<mi or x>ma:
        x = np.random.normal(a, max(ns/3,0)) 
    return x        

class Annotator:
    
    def __init__(self, experience, id):
        self.id = id 
        self.exp = experience
        noises = self.new_annotator(experience)
        self.noises = noises
        self.history = []
        self.history.append(self.noises)
        self.x, self.y, self.sx, self.sy = noises
        # self.conf = demi_gaussian(experience)
        
    def review(self, labels, org_label):
        
        return org_label, 0, 0 
        
        if org_label.shape[0]>0:
            org = org_label.cpu().numpy()  
            bias = self.noises
            evaluation = [] ; std = []
            for label in labels:
                std.append((org[:,1:] - np.array(label)[:,1:])**2)
                evaluation.append(np.mean(np.mean(std[-1], axis=1)))
                
            evaluation = np.array(evaluation)/np.sum(evaluation)
            # print(std, evaluation)   
            err = np.mean(np.array(std), axis=-1)
            # err = err/err.sum()  
            # err = (evaluation - err)**2
            
            choice = np.argmax(evaluation) # np.argmin(evaluation)
            # print(choice) 
             
            return labels[choice], evaluation[choice], err 
        return org_label, 0, 0 
             
    def evolve(self, incentive, mode="random", err=0, n_lvl=1):
        
        if mode=="random":
            mode = self.random_noise_reduction  
        else:
            mode = self.random_noise_reduction
            
        x, y, sx, sy = mode(self.noises, incentive*5, err, n_lvl)    
        self.noises = np.array([x, y, sx, sy])
        self.history.append(self.noises)
        self.x, self.y, self.sx, self.sy = self.noises
        self.exp = 1 - 0.25*sum(self.noises)
        # print(self.noises)
        
        return True
            
    def random_noise_reduction(self, noises, incentive, err=0, n_lvl=1):
        
        # incentive *= 4
        incentive = min(1,incentive)
        # random perfs for each type of noise
        red = n_lvl*(np.random.rand(4)*0.99 + 0.01) #  + (1-n_lvl)*err 
        # rescales them to match experience
        red /= red.sum()
        red = incentive * red
        red = (1-red)*noises # maintain the max reduction value available noise  
        
        return red
     
    def generate_a_label(self,new_lab): 
        # print(new_lab.cpu().numpy().shape)
        # box center coordinates
        """
        noise = noo(0,self.x,-0.5,0.5)  
        new_lab[-4] = max(0, min(1, new_lab[-4] + noise*new_lab[-2] ))
        noise = noo(0,self.y,-0.5,0.5) 
        new_lab[-3] = max(0, min(1, new_lab[-3] + noise*new_lab[-1] ))
        """
        shift = self.x * (np.random.rand()*2 - 1) 
        new_lab[-4] = max(new_lab[-2]/2, min(1-(new_lab[-2]/2), new_lab[-4] + shift*new_lab[-2] ))
        shift = self.y * (np.random.rand()*2 - 1)
        new_lab[-3] = max(new_lab[-1]/2, min(1-(new_lab[-1]/2), new_lab[-3] + shift*new_lab[-1] ))
    
        # box dimensions
        scale = self.sx * (np.random.rand()*1.5 + 0.5)
        nearest_border = min(1-new_lab[-4], new_lab[-4])
        new_lab[-2] = max(0, min(nearest_border*2, new_lab[-2]*scale))
        scale = self.sy * (np.random.rand()*1.5 + 0.5)
        nearest_border = min(1-new_lab[-3], new_lab[-3])  
        new_lab[-1] = max(0, min(nearest_border, new_lab[-1]*scale))
        
        # class
        # print(self.exp)
        if np.random.rand()>self.exp:
            # print(new_lab[-5], file=open("/auto/home/users/d/a/darimez/wtf.txt","a")) list(np.arange(15))
            choices = list(np.arange(15))
            choices.pop(int(new_lab[-5]))
            new_lab[-5] = np.random.choice(choices) # .pop(int(new_lab[-5]))) 
            """
            new_lab[-5] = new_lab[-5] + 1 if new_lab[-5]<14 else 0
            
            ['plane', 'ship', 'storage-tank', 'baseball-diamond', 'tennis-court', 'basketball-court', 'ground-track-field', 
               'harbor', 'bridge', 'large-vehicle', 'small-vehicle', 'helicopter', 'roundabout', 'soccer-ball-field', 'swimming-pool']
            
            class_ = new_lab[-5]
            if class_==0: # plane -> helicopter
                new_lab[-5] = 11    
            elif class_==1: # ship -> small, helicopter
                new_lab[-5] = np.random.choice([10, 11]) 
            elif class_==2: # storage -> large
                new_lab[-5] = 9 
            elif class_==3: # baseball -> [roundabout, pool,storage] 
                new_lab[-5] = np.random.choice([2,12,14]) 
            elif class_==4: # tennis -> [basket, soccer]
                new_lab[-5] = np.random.choice([5, 13]) 
            elif class_==5: # basket -> [tennis, soccer]
                new_lab[-5] = np.random.choice([4, 13]) 
            elif class_==6: # ground -> [harbor, bridge]
                new_lab[-5] = np.random.choice([7, 8]) 
            elif class_==7: # harbor -> [bridge, ground]
                new_lab[-5] = np.random.choice([6, 8]) 
            elif class_==8: # bridge -> [harbor, ground, large]
                new_lab[-5] = np.random.choice([6, 7, 9]) 
            elif class_==9: # large -> [bridge, storage]
                new_lab[-5] = np.random.choice([2, 8]) 
            elif class_==10: # small -> [large, pool, ship]
                new_lab[-5] = np.random.choice([9, 14, 1]) 
            elif class_==11: # helicopter -> plane
                new_lab[-5] = 0 
            elif class_==12: # round -> [soccer, storage, pool]
                new_lab[-5] = np.random.choice([11, 2, 14]) 
            elif class_==13: # soccer -> [tennis, basket]
                new_lab[-5] = np.random.choice([4, 5]) 
            else:            # pool -> [tennis, storage, roundabout, basket]
                new_lab[-5] = np.random.choice([4, 2, 12, 5]) 
            """
        return new_lab
     
    def generate_labels(self, olabs):
        if isinstance(olabs,np.ndarray): olabs = torch.from_numpy(olabs)
        new_labs = olabs.clone()  
        # print(new_labs.cpu().numpy().shape)
        if len(olabs)>1: 
            if len(olabs[0])>0: # several labels
                for new_i, new in enumerate(Parallel(n_jobs=-1, timeout=6000, pre_dispatch="2*n_jobs", verbose=0)(
                                                delayed(self.generate_a_label)(new_lab) for new_lab in olabs
                                            ) ) :
                    new_labs[new_i] = new  
            else: # 1 label
                new_labs[0] = self.generate_a_label(new_labs[0])
        return new_labs 
    
    def generate_labels_serial(self, olabs): 
        if isinstance(olabs,np.ndarray): olabs = torch.from_numpy(olabs)
        new_labs = olabs.clone()  
        if len(olabs)>0:   
            io = 0
            while io<len(olabs) if len(olabs[0])>0 else io<1: 
              
                new_lab = new_labs[io] if len(olabs[0])>0 else new_labs 
                generate_a_label(new_lab)
                
                if len(olabs[0])>0:
                    new_labs[io] = new_lab  
                else: 
                    new_labs = new_lab
                
                io += 1 
                 
        return new_labs
     
    def new_annotator(self, experience=0):
        seed = np.random.RandomState(123456)
        experience *= 4
        # random perfs for each type of noise
        noises = seed.rand(4) 
        # rescales them to match experience
        noises = (4 - experience) * noises/noises.sum()
        # maintain the max value to 1
        while np.any(noises>1):
            noises[noises>1] = 1
            missing = 4 - experience - noises.sum()
            new = seed.rand(np.sum(noises<1))
            noises[noises<1] += missing * new/new.sum()
        return noises