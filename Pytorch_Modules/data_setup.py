"""
creates PyTorch DataLoaders for images in a directory
structred as torch.datasets.ImageFolder.
"""
import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def create_dataloaders(
    train_dir: str, 
    test_dir: str, 
    train_transform: transforms.Compose,
    test_transform: transforms.Compose, 
    batch_size: int, 
    num_workers: int=os.cpu_count()
):
  """
  Creates training and testing DataLoaders.
  Returns:
    A tuple of (train_dataloader, test_dataloader, class_names)
  """
  train_data = datasets.ImageFolder(train_dir, transform=train_transform)
  test_data = datasets.ImageFolder(test_dir, transform=test_transform)

  class_names = train_data.classes

  train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True,
  )
  test_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      pin_memory=True,
  )
  
  return train_dataloader, test_dataloader, class_names