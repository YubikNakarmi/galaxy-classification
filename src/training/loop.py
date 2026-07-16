from __future__ import annotations
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path

root = Path(__file__).parent.parent.parent

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    criterion: nn.Module,
    writer: SummaryWriter,
    epoch: int,
    num_epochs: int,
    global_step: int,
    grad_clip: float,
    device: torch.device) -> tuple[float, int]:
    
    '''
        Run one training epoch.
    
        Returns:
            (train_loss, updated_global_step)
    '''
 
    model.train()
    running_loss = 0.0
    num_samples = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")

    for images,labels in pbar:
         
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)# Gradient clipping to prevent exploding gradients

        optimizer.step()
        scheduler.step(epoch + global_step/len(dataloader))  # type: ignore[arg-type]
        num_samples += batch_size
        global_step += 1

        #tensorboard loggings
        writer.add_scalar("Loss/train", loss.item(), global_step)
        writer.add_scalar("LearningRate", optimizer.param_groups[0]['lr'], global_step)
        
        #cli visual
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
        })

    train_loss = running_loss / num_samples
    return train_loss, global_step

def validate_epoch(
     model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    metrics: dict,
    epoch: int,
    num_epochs: int,
    device: torch.device) ->  tuple[float, dict]:

    ''' 
    Run one validation epoch.

    Returns:
        (val_loss, metrics_results)
    
    '''

    running_loss = 0.0
    num_samples = 0


    model.eval()
    val_loss = 0.0
    with torch.no_grad():

        for images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [val]"):
            images, labels = images.to(device), labels.to(device)

            
            outputs = model(images)
            loss = criterion(outputs, labels)
            batch_size = images.size(0)

            running_loss += loss.item() * batch_size
            num_samples += batch_size

            preds = torch.softmax(outputs, dim=1)

            for m in metrics.values():#update at each batch
                m.update(preds, labels)

        # compute loss once, then build results from it
        val_loss = running_loss / num_samples if num_samples > 0 else 0.0

        results = {
        "val_loss":             val_loss,
        "val_acc":              metrics["accuracy"].compute().item(),
        "val_f1_micro":         metrics["f1_micro"].compute().item(),
        "val_f1_macro":         metrics["f1_macro"].compute().item(),
        "val_f1_classwise":     metrics["f1_classwise"].compute().tolist(),
        "val_precision_micro":  metrics["precision_micro"].compute().item(),
        "val_precision_macro":  metrics["precision_macro"].compute().item(),
        "val_precision_classwise": metrics["precision_classwise"].compute().tolist(),
        "val_recall_micro":     metrics["recall_micro"].compute().item(),
        "val_recall_macro":     metrics["recall_macro"].compute().item(),
        "val_recall_classwise": metrics["recall_classwise"].compute().tolist(),
        "val_roc_auc":          metrics["roc_auc"].compute().item(),
        }
        
        for m in metrics.values():
            m.reset()

        return val_loss, results
