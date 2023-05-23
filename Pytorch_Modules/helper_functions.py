"""
Contains functions for data preparation, evaluation, visualizing.
"""
import os
from pathlib import Path
from google.colab import drive
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Dict, List
from tqdm.auto import tqdm
import torchmetrics, mlxtend
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

def create_dataset_dir(classes_names:list[str], dataset_path:Path):
    train_test = ["train", "test"]
    if dataset_path.is_dir():
      print("found")
    else:
      print("Created new dataset directory")
    for directory in iter(train_test):
        for class_ in iter(classes_names):
          target = dataset_path / directory / class_
          target.mkdir(parents=True, exist_ok=True)
    walk_through_dir(dataset_path)
    
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


def plot_loss_curves(results: Dict[str, List[float]]):
  loss = results["train_loss"]
  test_loss = results["test_loss"]

  accuracy = results["train_acc"]
  test_accuracy = results["test_acc"]

  # Figure out the count od epocs
  epochs = range(len(results["train_loss"]))

  plt.figure(figsize=(15, 7))
  plt.subplot(1, 2, 1)
  plt.plot(epochs, loss, label="train_loss")
  plt.plot(epochs, test_loss, label="test_loss")
  plt.title("Loss")
  plt.xlabel("Epochs")
  plt.legend()

  plt.subplot(1, 2, 2)
  plt.plot(epochs, accuracy, label="train_acc")
  plt.plot(epochs, test_accuracy, label="test_acc")
  plt.title("Accuracy")
  plt.xlabel("Epochs")
  plt.legend()


def predict_on_custom(model: torch.nn.Module, 
                      img_path: str,
                      device: torch.device, 
                      class_names: List[str] = None, 
                      target_size=None):
  """a function to predict on a custom image and plot the image with pred prop"""
  with Image.open(img_path) as img:
    rgb_image = img.convert('RGB')
    custom_transform = transforms.Compose([transforms.Resize(size=(target_size, target_size)),
                                           transforms.ToTensor()])
    transformed_image = custom_transform(rgb_image)

    plt.imshow(rgb_image)

  model.eval()

  with torch.inference_mode():
    preds = model(transformed_image.unsqueeze(dim=0).to(device))

  target_image_pred_probs = torch.softmax(preds, dim=1)
  class_idx = torch.argmax(target_image_pred_probs, dim=1)

  if class_names:
      title = f"Pred: {class_names[class_idx.cpu()]} | Prob: {target_image_pred_probs.max().cpu():.3f}"
  else: 
      title = f"Pred: {class_idx} | Prob: {target_image_pred_probs.max().cpu():.3f}"

  plt.title(title)
  plt.axis(False)

def make_predictions(dataloder:torch.utils.data.DataLoader, model_:torch.nn.Module,
                      device: torch.device):
  """ raturns: preds_tensor, true_predictions"""
  # 1. Make predictions with trained model
  y_preds = []
  y_targets = []  # to save the targets with the same shuffle, in case shuffle is true in the dataloader
  model_.eval()
  with torch.inference_mode():
    for X, y in tqdm(dataloder, desc="Making predictions"):
      # Send data and targets to target device
      X, y = X.to(device), y.to(device)
      # Do the forward pass
      y_logit = model_(X)
      y_pred = y_logit.argmax(dim=1)
      # Put predictions on CPU for evaluation
      y_preds.append(y_pred.cpu())
      y_targets.append(y.cpu())
  # Concatenate list of predictions into a tensor
  preds_tensor = torch.cat(y_preds)
  true_predictions = torch.cat(y_targets)
  return preds_tensor, true_predictions

def plot_confusion(targets, predictions, class_names):
  confmat = ConfusionMatrix(num_classes=len(class_names), task='multiclass')
  confmat_tensor = confmat(preds=predictions,
                          target=targets)

  # 3. Plot the confusion matrix
  fig, ax = plot_confusion_matrix(
      conf_mat=confmat_tensor.numpy(), # matplotlib likes working with NumPy 
      class_names=class_names, # turn the row and column labels into class names
      figsize=(10, 7)
  );

def plot_loss_curves(results: Dict[str, List[float]]):
  loss = results["train_loss"]
  test_loss = results["test_loss"]

  accuracy = results["train_acc"]
  test_accuracy = results["test_acc"]

  # Figure out the count od epocs
  epochs = range(len(results["train_loss"]))

  plt.figure(figsize=(15, 7))
  plt.subplot(1, 2, 1)
  plt.plot(epochs, loss, label="train_loss")
  plt.plot(epochs, test_loss, label="test_loss")
  plt.title("Loss")
  plt.xlabel("Epochs")
  plt.legend()

  plt.subplot(1, 2, 2)
  plt.plot(epochs, accuracy, label="train_acc")
  plt.plot(epochs, test_accuracy, label="test_acc")
  plt.title("Accuracy")
  plt.xlabel("Epochs")
  plt.legend()

def compare_loss_curves(results_1: Dict[str, List[float]],
                        results_2: Dict[str, List[float]]):
  # Setup a plot
  plt.figure(figsize=(15, 10))
  epochs = range(len(results_1["train_loss"]))

  # plot train loss
  plt.subplot(2, 2, 1)
  plt.plot(epochs, results_1["train_loss"], label="model 0")
  plt.plot(epochs, results_2["train_loss"], label="model 1")
  plt.title("Train loss")
  plt.xlabel("Epochs")
  plt.legend()

  # plot test loss
  plt.subplot(2, 2, 2)
  plt.plot(epochs, results_1["test_loss"], label="model 0")
  plt.plot(epochs, results_2["test_loss"], label="model 1")
  plt.title("Test loss")
  plt.xlabel("Epochs")
  plt.legend()

  # plot train acc
  plt.subplot(2, 2, 3)
  plt.plot(epochs, results_1["train_acc"], label="model 0")
  plt.plot(epochs, results_2["train_acc"], label="model 1")
  plt.title("train acc")
  plt.xlabel("Epochs")
  plt.legend()

  # plot test acc
  plt.subplot(2, 2, 4)
  plt.plot(epochs, results_1["test_acc"], label="model 0")
  plt.plot(epochs, results_2["test_acc"], label="model 1")
  plt.title("Test acc")
  plt.xlabel("Epochs")
  plt.legend()


