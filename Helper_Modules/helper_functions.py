"""
Contains functions for data preparation, evaluation, visualizing.
"""
import os
from pathlib import Path
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from typing import Dict, List
from tqdm.auto import tqdm
from Helper_Modules.resize import to_square_save

def set_seeds(seed: int=42):
  # torch.backends.cudnn.deterministic = True
  random.seed(42)
  torch.manual_seed(42)
  torch.cuda.manual_seed_all(42)


def predict_on_list(transforms, model, device, class_names, paths_list, figsize=(15,30), to_suare_=False):
    """width gotta be 15 (15, ..)\n
    args:\ntransforms:[torchvision.transforms]\n
    model: torch.nn.Module -> the model to make predictions with\n
    device: torch.device\n
    class_names:[str] -> with the same lables order\n
    paths_list: [str]\n
    figsize: (int,int)\n
    to_square -> should the images be cropped or not
    """
    # height = math.ceil(len(paths_list)/5) * 5 TODO: look for another method 
    plt.figure(figsize=figsize)
    index = 1
    for path in paths_list:
        if index < 61:
          plt.subplot(12, 5, index)
          predict_on_custom(model, path, device, class_names, to_square_=to_suare_, transformations=transforms)
          index+=1
        else:
           break


def predict_on_custom(model: torch.nn.Module, 
                      img_path: str,
                      device: torch.device, 
                      class_names: List[str] = None, 
                      to_square_=False,
                      plot=True,
                      transformations=None):
  """a function to predict on a custom image and plot the image with pred prop"""
  if to_square_:
     img = to_square_save(img_path)
  else:
     img = Image.open(img_path)

  rgb_image = img.convert('RGB')
  custom_transform = transforms.Compose(transformations)
  transformed_image = custom_transform(rgb_image)

  model.eval()

  with torch.inference_mode():
    preds = model(transformed_image.unsqueeze(dim=0).to(device))

  target_image_pred_probs = torch.softmax(preds, dim=1)
  class_idx = torch.argmax(target_image_pred_probs, dim=1)
  max_prop = target_image_pred_probs.max().cpu()

  if class_names:
      title = f"Pred: {class_names[class_idx.cpu()]} | Prob: {max_prop:.3f}"
  else: 
      title = f"Pred: {class_idx} | Prob: {max_prop:.3f}"
  if plot:
    plt.imshow(rgb_image)
    plt.title(title)
    plt.axis(False)
  return class_idx, max_prop

# all the upcoming functions have been used earlier, and they can be useful for later work

def walk_through_dir(dir_path):
  for dirpath, dirnames, filenames in os.walk(dir_path):
      print(f"Found {len(dirnames)} directories and {len(filenames)} Images in {dirpath}")

def get_random_image_from_dir(dataset_path):
  image_path_list = list(dataset_path.glob("*/*/*.png"))
  random_image_path = random.choice(image_path_list)
  image_class = random_image_path.parent.stem
  img = Image.open(random_image_path)

  print(f"Random image path: {random_image_path}")
  print(f"Image Class: {image_class}")
  print(f"Image height: {img.height}")
  print(f"Image width: {img.width}")
  return img, random_image_path

def plot_transformed_images(image_paths, transform, n=2):
  """ A fuction to plot the transformed images and compare them to original 
  """
  random_image_paths = random.sample(image_paths, k=n)
  for image_path in random_image_paths:
    with Image.open(image_path) as f:
      fig, ax = plt.subplots(1, 2)  # one row and two cols ax=coloumn
      ax[0].imshow(f)
      ax[0].set_title(f"Original\nSize:{f.size}")
      ax[0].axis("off")

      # transform and plot image
      # The transformed Tensor has the shape CHW while matplotlib perfers HWC
      # this why permute the tensor (1, 2, 0)
      transformed_image = transform(f).permute(1, 2, 0)
      ax[1].imshow(transformed_image)
      ax[1].set_title(f"Transformed\nSize{transformed_image.shape}")
      ax[1].axis("off")

      fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=10)


