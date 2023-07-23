"""
This script contains functions to initiate specific models(more will be added),
and some functions to log the models and chaneg them 
"""

import torch
import wandb
import torchvision

def efficientNet_b0(freeze_layers, num_classes, device, transfer=True):
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    if transfer:
        model = torchvision.models.efficientnet_b0(weights=weights).to(device)
    else:
        model = torchvision.models.efficientnet_b0().to(device)

    if freeze_layers:
        print("freezing")
        for param in model.features.parameters():
            param.requires_grad = False

    model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True), 
    torch.nn.Linear(in_features=1280, 
                    out_features=num_classes, # same number of output units as our number of classes
                    bias=True)).to(device)
    
    return model

def build_model_and_log(model, config):
    """Intializes a new model and logs it to WandB"""
    with wandb.init(project=config['project_name'], job_type="initialize-model", config=config) as run:
        
        model_artifact = wandb.Artifact(
            config['model_name'], type="model",
            description=config['description'])
        
        torch.save(model.state_dict(), "initialized_model.pth")
       
        model_artifact.add_file("initialized_model.pth")

        wandb.save("initialized_model.pth")

        run.log_artifact(model_artifact)
    return model
