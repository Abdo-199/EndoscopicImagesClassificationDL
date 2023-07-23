import torch
import wandb
from torch.utils.data import TensorDataset
import os
import platform
## using os path to be independent of the OS 

def read(data_dir, split):
    """Reads a .pt file and returns a TensorDataset"""
    filename = split + ".pt"
    x, y = torch.load(os.path.join(data_dir, filename))

    return TensorDataset(x, y)

def load_data_wandb(dataload_config):
    """ dataload_config : {"datasetname":"","version":""}
    returns: TensorDatasets: train_dataset, test_dataset
    """
    if platform.system() == "Windows":
        data_path = os.path.join("artifacts", "datasets", replace_last_occurrence(dataload_config['dataset_name'], ':', '-'))
    else:
        data_path = os.path.join("artifacts", "datasets", dataload_config['dataset_name'])



    if dataload_config['download'] or not os.path.exists(data_path):
        run = wandb.init()
        artifact = run.use_artifact(f"d-ml/{dataload_config['project_name']}/{dataload_config['dataset_name']}",
                                    type='dataset')
        artifact_dir = artifact.download()
        data_path = os.path.join("artifacts", "datasets", os.path.split(artifact_dir)[-1])

        if not os.path.exists(os.path.join("artifacts", "datasets")):
            print("creating datasets dir")
            os.makedirs(os.path.join("artifacts", "datasets"))

        os.replace(artifact_dir, data_path)

    train_dataset =  read(data_path, "train")
    test_dataset = read(data_path, "test")
    return train_dataset, test_dataset

    # with wandb.init(project=dataload_config['project_name'], config=dataload_config,job_type="load-data") as run:
        
    #     raw_data_artifact = run.use_artifact(dataload_config['dataset_name'])

    #     data_dir = raw_data_artifact.download()
    #     new_dir = os.path.join("artifacts", "datasets", data_dir.split("/")[-1])
    #     os.replace(data_dir, new_dir)
        
    
def download_model(project_name, model_name, file_name, model):

    if platform.system() == "Windows":
        model_path = os.path.join("artifacts", "models", replace_last_occurrence(model_name, ':', '-'), file_name)
    else:
        model_path = os.path.join("artifacts", "models", model_name, file_name)
           
    if not os.path.exists(model_path):
        print(f"downloading the model: {model_name}")
        run = wandb.init()
        artifact = run.use_artifact(f'd-ml/{project_name}/{model_name}', type='model')
        artifact_dir = artifact.download()
        new_model_dir = os.path.join("artifacts",  "models", os.path.split(artifact_dir)[-1])

        if not os.path.exists(os.path.join("artifacts", "models")):
            print("creating models dir")
            os.makedirs(os.path.join("artifacts", "models"))
      
        os.replace(artifact_dir, new_model_dir)
        model_path = os.path.join(new_model_dir, file_name) # just to make sure (in case of latest)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return model

def replace_last_occurrence(string, old, new):
    """replaces the last occurence of a char in a string.
    used to enable the usage of the code in linux and windows"""
    reversed_string = string[::-1]
    reversed_old = old[::-1]
    reversed_new = new[::-1]

    replaced = reversed_string.replace(reversed_old, reversed_new, 1)
    result = replaced[::-1]

    return result

