"""
train and log into wandb
"""
import torch
from torch.utils.data import DataLoader
import wandb
import os
import platform
import shutil
from Helper_Modules.utils import read
from Helper_Modules.train_test import train
from Helper_Modules.utils import replace_last_occurrence
from Helper_Modules.helper_functions import set_seeds

def train_and_log(model, train_config):
    """
    A function to train a model and log the results into WandB\n
    training_dict = dict(project_name -> project name in WandB,\n
                     artifact_name -> trained model name to be logged into WandB,\n
                     artifact_discription -> description of the result model,\n
                     dataset -> dataset to be used from WandB,\n
                     download_data -> download the dataset from WandB,\n
                     run_number -> the run number in the script,\n
                     batch_size,\n
                     model_name -> input model name in WandB",\n
                     model_filename -> the file name of the input model,\n
                     continue_training -> should the state dict of the input model be loaded?,\n
                     optimizer=torch.optim.Adam(params=model_.parameters(),lr=),\n
                     loss_fn=torch.nn.CrossEntropyLoss(),\n
                     epochs,\n
                     class_names=[string],\n
                     device=device)
    """
    set_seeds()
    if not os.path.exists(os.path.join("artifacts", "models")):
            print("creating models dir")
            os.makedirs(os.path.join("artifacts", "models"))
    if not os.path.exists(os.path.join("artifacts", "datasets")):
            print("creating datasets dir")
            os.makedirs(os.path.join("artifacts", "datasets"))
            
    with wandb.init(project=train_config['project_name'], job_type="train", config=train_config) as run:
        config = wandb.config
        
        # in windows dir names can't contain ':'
        if platform.system() == "Windows":
            dataset_path = os.path.join(".\\artifacts", "datasets", replace_last_occurrence(train_config['dataset'], ':', '-'))
            model_path = os.path.join( "artifacts", "models",
                                  replace_last_occurrence(train_config['model_name'], ':', '-'), train_config['model_filename'])
        else:
            dataset_path = os.path.join("artifacts", "datasets", train_config['dataset'])
            model_path = os.path.join( "artifacts", "models",
                                  train_config['model_name'], train_config['model_filename'])
         
        data = run.use_artifact(train_config['dataset'])

        if train_config['download_data'] or not os.path.exists(dataset_path):
            print(f"downloadinng the dataset: {train_config['dataset']}")
            download_dir = data.download()
            data_dir = os.path.join("artifacts", "datasets", os.path.split(download_dir)[-1])
            os.replace(download_dir, data_dir)
        else:
            data_dir =  dataset_path

        train_dataset =  read(data_dir, "train")
        test_dataset = read(data_dir, "test")

        train_loader = DataLoader(train_dataset,
                                batch_size=train_config['batch_size'],
                                shuffle=True,
                                num_workers=os.cpu_count(),
                                pin_memory=True,
                                )
        test_loader = DataLoader(test_dataset,
                                batch_size=train_config['batch_size'],
                                shuffle=False,
                                num_workers=os.cpu_count(),
                                pin_memory=True,
                                )

        model_artifact = run.use_artifact(train_config['model_name'])
        model_path = os.path.join( "artifacts", "models",
                                  train_config['model_name'], train_config['model_filename'])
        # in case of continuing training
        if train_config['continue_training']:
            if not os.path.exists(model_path):
               print(f"downloading the model: {train_config['model_name']}")
               model_dir = model_artifact.download() 
               new_model_dir = os.path.join("artifacts",  "models", os.path.split(model_dir)[-1])
               os.replace(model_dir, new_model_dir)
               model_path = os.path.join(new_model_dir, train_config['model_filename']) # just to make sure (in case of latest)
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        
        model_config = model_artifact.metadata
        config.update(model_config)
        model = model.to(train_config['device'])
 
        results, preds_, labels_l, props_ = train(model=model,
                                                train_dataloader= train_loader,
                                                test_dataloader= test_loader,
                                                optimizer= train_config['optimizer'],
                                                loss_fn=train_config['loss_fn'],
                                                epochs=train_config['epochs'],
                                                class_names=train_config['class_names'],
                                                device=train_config['device'])

        model_artifact = wandb.Artifact(
            train_config['artifact_name'], type="model",
            description=train_config['artifact_discription'])

        torch.save(model.state_dict(), "trained_model.pth")
        model_artifact.add_file("trained_model.pth")
        wandb.save("trained_model.pth")

        run.log_artifact(model_artifact)

    return model, results, preds_, labels_l, props_