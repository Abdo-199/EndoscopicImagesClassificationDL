"""
This script contains the train and test logic to use with WandB
"""
import torch
import wandb
from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch.

    Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy).
    """
    model.train()

    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = (train_acc / len(dataloader)) * 100
    return train_loss, train_acc



def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device) :
    """Tests a PyTorch model for a single epoch.
    Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (test_loss, test_accuracy, Images).
    """
    model.eval() 

    test_loss, test_acc = 0, 0
    preds_ = []
    preds_idx = []
    labels_ = []
    props_ = []

    with torch.inference_mode():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            test_pred_logits = model(images)

            loss = loss_fn(test_pred_logits, labels)
            test_loss += loss.item()
            target_image_pred_probs = torch.softmax(test_pred_logits, dim=1)
            test_pred_idx = torch.argmax(target_image_pred_probs, dim=1)

            props_.append(target_image_pred_probs)
            preds_.append(test_pred_logits.tolist())
            preds_idx.append(test_pred_idx.tolist())
            labels_.append(labels.tolist())
            test_acc += ((test_pred_idx == labels).sum().item()/len(test_pred_idx))

    props_ = [item for sublist in props_ for item in sublist]
    preds_ = [item for sublist in preds_ for item in sublist]
    preds_idx = [item for sublist in preds_idx for item in sublist]
    labels_ = [item for sublist in labels_ for item in sublist]  # flatten the lists to enable using the confmat
    test_loss = test_loss / len(dataloader)
    test_acc = (test_acc / len(dataloader)) * 100

    return  test_loss, test_acc, preds_, preds_idx, labels_, props_

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          class_names,
          device: torch.device) -> Dict[str, List]:
    """
    trains a model
    Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for 
    each epoch.
    For example if training for epochs=2: 
             {train_loss: [1.5934, 0.9675],
              train_acc: [40.45, 50.45],
              test_loss: [1.9746, 1.1234],
              test_acc: [40.00, 45.73]} 
    """
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []
    }
    wandb.watch(model, loss_fn, log="all", log_freq=10)
    model.to(device)

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          device=device)
        test_loss, test_acc, preds_, preds_idx, labels_l, props_ = test_step(model=model,
          dataloader=test_dataloader,
          loss_fn=loss_fn,
          device=device)

        print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.2f}% | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.2f}%"
        )
        wandb.log({"epoch": epoch, "train_loss": train_loss}, step=epoch)
        wandb.log({"epoch": epoch, "train_acc": train_acc}, step=epoch)
        wandb.log({"epoch": epoch, "test_loss": test_loss}, step=epoch)
        wandb.log({"epoch": epoch, "test_acc": test_acc}, step=epoch)
        

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None,y_true=labels_l
                                                        ,preds=preds_idx,
                                                        class_names=class_names)})
    if len(class_names) == 2:
      wandb.log({"roc" : wandb.plot.roc_curve( labels_l, preds_, labels="good",
                        classes_to_plot=1)})

    return results, preds_, labels_l, props_
