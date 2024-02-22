from train import * 
from create_dataset import create_kd_dataset 
 
@hydra.main(version_base=None, config_path=f"..{os.sep}conf", config_name="config")
def main(cfg: DictConfig) -> None:

    base_path = Path(cfg['root'])    
    name = f"{uuid.uuid4().hex.upper()[0:6]}_{cfg['name']}"
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
                     compute_norm=cfg["compute_norm"] ) 
    model.train() # alrounds=cfg["active"]["rounds"]) 
    

if __name__ == '__main__':
    main()