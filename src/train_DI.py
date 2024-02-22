import sys, os
sys.path.append(os.path.join(os.getcwd(), "ultralytics")) 

import copy
import yaml, glob
import wandb
import numpy as np
import shutil
from shutil import copyfile

import uuid
from pathlib import Path 
import random
from ultralytics import YOLO
sys.path.append(os.path.join(os.getcwd(), "src")) 
from decoder import YOLODecoder

def create_mixte_dataset(images_dir = "", 
                         txt_dir = "",
                         data_size = 1000, 
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
      
    images = list_images(images_dir, formats, val + test + data_size, None, keep_backgrouds=keep_backgrouds)
    train_images = sorted(images)

    # shuffle images
    random.Random(42).shuffle(train_images) 
 
    train_txt_path = txt_dir / 'train.txt'
    unlab_txt_path = txt_dir / 'unlabelled.txt'
    val_txt_path = txt_dir / 'val.txt'
    test_txt_path = txt_dir / 'test.txt'
    data_yaml_path = txt_dir / 'data.yaml'
    
    this_train_images = train_images[:data_size]  
    val_images = train_images[data_size:data_size+val]  
    test_images = train_images[data_size+val:data_size+val+test]
    
    with open(train_txt_path, 'w') as f:
        f.write('\n'.join(this_train_images))  
    with open(val_txt_path, 'w') as f:
        f.write('\n'.join(val_images)) 
    with open(test_txt_path, 'w') as f:
        f.write('\n'.join(test_images))

    create_yaml_file(data_yaml_path, train_txt_path, val_txt_path, test_txt_path, classes)
     
     
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
    images_path = Path(images_path) / 'images'
    for format in formats:
        images += [
            *glob.glob(str(images_path.absolute()) + f'/*.{format}')
        ]
    images = filter_classes(images, classes_to_filter, keep_backgrouds, formats=formats)
    return images 

def create_yaml_file(save_path: Path, train: Path, val: Path, test: Path, classes=[]):
    """
    Construct the yaml file

    :param pathlib.Path txt_dir: path used to create the txt files
    :param pathlib.Path yaml_dir: path used to create the yaml file

    :return: None
    :rtype: NoneType
    """
    yaml_file = {
        'train': str(train.absolute()), 
        'val': str(val.absolute()),
        'test': str(test.absolute()),
        'names': {ii: class_name for ii, class_name in enumerate(classes)}
    }

    with open(save_path, 'w') as file:
        yaml.dump(yaml_file, file)   
    
                  
if __name__ == '__main__': 
    ######## Creates Dataset 
    yaml_path = "E:/FedYOLO/data/coco/"
    create_mixte_dataset(images_dir="E:/FedYOLO/data/coco/", 
                         txt_dir=str(yaml_path),
                         data_size=640, 
                         val=320,
                         test=320)  
                                 
    ####### Creates Model
    decoder = YOLODecoder()
    model = YOLO("yolov8s.yaml")
    
    ######## Train model
    model.train(name="yolo",
                entity="dany404",
                project="DI",
                data=yaml_path+'/data.yaml', 
                epochs=200, 
                lr0=0.01, 
                lrf=0.01,
                amp=False,
                imgsz=320, 
                batch=32, 
                nbs=64, 
                decoder=decoder)
 
 
 
