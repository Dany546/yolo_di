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
    print("selected", selected); used_data = list(U) + selected
    
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
        data_yaml_file["train"] = train; print("train it==1", train)
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
     
      