"""
This script contains functions for loading the datasets into and from Wandb, 
and process it and load it again.
"""
import torch
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset
from collections import Counter
import wandb
from Helper_Modules.utils import read, replace_last_occurrence
from Helper_Modules.helper_functions import predict_on_custom
import os
import platform
from pathlib import Path
import shutil

def convert_to_tensorDataset(dataset):
    """ converts a ImageFolder dataset to a TensorDataset"""
    images, labels = [], []
    for image, label in dataset:
        images.append(image)
        labels.append(label)
    images = torch.stack(images)
    labels = torch.tensor(labels)
    tensor_dataset = TensorDataset(images,labels)

    return tensor_dataset

def get_data_dict(split_name, dataset):
    """Gets the count of the images in each class in a dataset. *No TensorDataset*
    split_name: test or train.."""
    dataset.class_to_idx
    idx_to_class = {v:k for k, v in dataset.class_to_idx.items()}  
    idx_distribution = dict(Counter(dataset.targets))
    class_distribution = {idx_to_class[k]:v for k, v in idx_distribution.items()} | {"sum": len(dataset)}

    dicto = {split_name: class_distribution}
    return dicto

def load_ImageFolder(dataset_dir:str, transforms_):
    """loads an ImageFolder: size(int,int)
    returns: List[TensorDataset], Classes Dictionary"""

    train_data = datasets.ImageFolder(f"{dataset_dir}/train", transform=transforms.Compose(transforms_))
    test_data  = datasets.ImageFolder(f"{dataset_dir}/test" , transform=transforms.Compose(transforms_))
    dicto_1 = get_data_dict("train_dataset", train_data)
    dicto_2 = get_data_dict("test_dataset", test_data)
    metadata = dicto_1 | dicto_2
    train_data = convert_to_tensorDataset(train_data)
    test_data = convert_to_tensorDataset(test_data)

    dataset = [train_data, test_data]
    return dataset, metadata

def load_and_log_IF(dataset_config):
    """Needs improvement(faster way to convert to TensorDataset)
    Loads an image folder as a dataset and logs it inot wandb
    dataset_config = dict(dataset_dir=dataset_path,
                      size=(224,224),
                      name="original_endo",
                      description="the raw version of the original dataset")"""
    with wandb.init(project=dataset_config['project_name'], config=dataset_config, job_type="load_data") as run:
        
        config = wandb.config

        datasets, metadata = load_ImageFolder(dataset_dir=dataset_config['dataset_dir'], transforms_=dataset_config['transforms_'])
        splits = ["train", "test"]
        
        artifact = wandb.Artifact(name=config.name,
                              type="dataset",
                              description=config.description,
                              metadata=metadata)
        
        for split, data in zip(splits, datasets):
            with artifact.new_file(split + ".pt", mode="wb") as file:
                X, y = data.tensors
                torch.save((X, y), file)

        run.log_artifact(artifact)

def apply_transforms(dataset, transforms_list: transforms.Compose):
    """ takes TensorDataset() and applies transforms on it
    transforms_list:[]
    Note: try to merge it with convert_to_tensor_ds()"""
    data_transform = transforms.Compose(transforms_list)
    images = []
    labels = []
    for sample, label in dataset:
        transformed_sample = data_transform(sample)
        images.append(transformed_sample)
        labels.append(label)

    images = torch.stack(images)
    labels = torch.tensor(labels)
    transformed_dataset = TensorDataset(images,labels)

    return transformed_dataset

def process_and_log(dataset_config):
    with wandb.init(project=dataset_config['project_name'], job_type="preprocess-data", config=dataset_config) as run:
        
        processed_data = wandb.Artifact(
            dataset_config['name'], type="dataset",
            description=dataset_config['description'])
         
        raw_data_artifact = run.use_artifact(dataset_config['input_dataset'])

        if platform.system() == "Windows":
            raw_dataset = os.path.join("artifacts", "datasets", replace_last_occurrence(dataset_config['input_dataset'], ':', '-'))
        else:
            raw_dataset = os.path.join("artifacts", "datasets", dataset_config['input_dataset'])

        if dataset_config['download'] or not os.path.exists(raw_dataset):
            print("downloading")
            download_dir = raw_data_artifact.download()
            os.makedirs(os.path.join("artifacts", "datasets"), exist_ok=True)
            raw_dataset = os.path.join("artifacts", "datasets",os.path.split(download_dir)[-1])
            os.replace(download_dir, raw_dataset)
        
        for split in ["train", "test"]:
            raw_split = read(raw_dataset, split)
            if split=="train" or dataset_config['apply_on_test']:
                processed_dataset = apply_transforms(raw_split, dataset_config['transforms'])
            else:
                processed_dataset = raw_split
                
            with processed_data.new_file(split + ".pt", mode="wb") as file:
                x, y = processed_dataset.tensors
                torch.save((x, y), file)

        run.log_artifact(processed_data)

def sort_out_noise(transforms, model, dir_name, device, class_names, paths_list, to_suare_=False, threshold=0.5):
  """classifies the quality of all images in in a dataset directory in form of:
  directory/organ/patient/images
  and divides them into two directories clean and noisy
  args:\n transforms: [torchvision.transforms]
  model: The quality classifier to be used to filter the images\n
  dir_name: name of the destination directory under ../data

  """
  for path in paths_list:
    class_idx, max_prop = predict_on_custom(model, path, device, class_names, to_square_=to_suare_, plot=False, transformations=transforms)
    if class_idx == 0 and max_prop > threshold:
      # just change the name of the dataset upper dir. pack the used propapility threshold for the bad images in the name 
      new_path = Path(os.path.join(path.parents[3], dir_name, f"{path.parts[-4]}-noisy-{threshold}", path.parts[-3], path.parts[-2], path.parts[-1]))
      os.makedirs(new_path.parents[0], exist_ok=True)
    else:
      new_path = Path(os.path.join(path.parents[3], dir_name, f"{path.parts[-4]}-clean-{threshold}", path.parts[-3], path.parts[-2], path.parts[-1]))
      os.makedirs(new_path.parents[0], exist_ok=True)
      
    shutil.copy(path, new_path)

