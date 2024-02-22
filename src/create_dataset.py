"""
Dataset 
"""
import yaml

def create_mixte_dataset(images_dir = "", 
                         txt_dir = "",
                         data_sizes = [1000, 1000, 1000, 1000, 1000],
                         labelled_perc = 0.1,
                         val = 1000,
                         test = 2000,
                         formats = ['jpg', 'png', 'jpeg']):
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
    images_path = Path(images_dir) / 'images'

    classes = [ 'plane', 'ship', 'storage-tank', 'baseball-diamond', 'tennis-court', 
                'basketball-court', 'ground-track-field', 'harbor', 'bridge', 'large-vehicle', 
                'small-vehicle', 'helicopter', 'roundabout', 'soccer-ball-field', 'swimming-pool' ]
    classes_to_filter = np.ones(len(classes))*1000000
    images = list_images(images_path, formats, val + test + sum(data_sizes), None, keep_backgrouds=True)
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

def create_kd_dataset(teachers, N_images, ref_data_yaml_path):
 
    with open(ref_data_yaml_path, "r") as yaml_file:
        new_yaml = yaml.safe_load(yaml_file)
    
    images = list_images(images_path, formats, val + test + sum(data_sizes), None, keep_backgrouds=True)
    val_images = images[:val]
    test_images = images[val:val+test]
    train_images = sorted(images[-N_images:])
    
    # shuffle images
    random.Random(42).shuffle(train_images) 

    txt_dir = Path("/".join(ref_data_yaml_path.split("/")[:-2]))
    id = kd
    
    train_txt_path = txt_dir / str(id) / 'train.txt' 
    val_txt_path = txt_dir / str(id) / 'val.txt'
    test_txt_path = txt_dir / str(id) / 'test.txt'
    data_yaml_path = txt_dir / str(id) / 'data.yaml'
    
    with open(train_txt_path, 'w') as f:
        f.write('\n'.join(train_images))  
    with open(val_txt_path, 'w') as f:
        f.write('\n'.join(val_images)) 
    with open(test_txt_path, 'w') as f:
        f.write('\n'.join(test_images))

    classes = [ 'plane', 'ship', 'storage-tank', 'baseball-diamond', 'tennis-court', 
                'basketball-court', 'ground-track-field', 'harbor', 'bridge', 'large-vehicle', 
                'small-vehicle', 'helicopter', 'roundabout', 'soccer-ball-field', 'swimming-pool' ] 
    create_yaml_file(data_yaml_path, train_txt_path, train_txt_path, val_txt_path, test_txt_path, classes)

    new_yaml["train"] =  
    

def filter_classes(images, classes_to_filter, keep_backgrouds=True, formats=[]):

    if classes_to_filter is None and keep_backgrouds:
        return images
    
    new_images = [] 
    for im in images:
        label = im.replace("images","labels")
        for format in formats:
            label = label.replace(f".{format}",".txt")
        label = np.loadtxt(label)
        if len(label)==0 and keep_backgrouds:
            new_images.append(im)
        elif classes_to_filter is None:
            new_images.append(im)
        elif len(label)>0:
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
        

def list_images(images_path: Path, formats=[], max_=None, classes_to_filter=[], keep_backgrouds=True):
    images = []
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
        
        