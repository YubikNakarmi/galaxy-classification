from __future__ import annotations
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from src.cli.style import UI,_TrainSession
root = Path(__file__).parent.parent.parent


from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    LinearLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    SequentialLR,
    LRScheduler,
)


def get_scheduler(
    optimizer: Optimizer, train_dataloader, cfg,) -> LRScheduler:

    steps_per_epoch = len(train_dataloader)
    total_steps = cfg.train.num_epochs * steps_per_epoch
    warmup_steps = cfg.train.scheduler.warmup_epochs * steps_per_epoch

    scheduler_type = cfg.train.scheduler.name

    #no restart with just warmup
    if scheduler_type == "cosine_warmup":

        warmup = LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)

        cosine = CosineAnnealingLR(
            optimizer, T_max=total_steps - warmup_steps, eta_min=cfg.train.scheduler.min_lr,)

        return SequentialLR( #combine into seq
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_steps])

    #linear + simple cosine annealing with restart
    elif scheduler_type == "cosine_warmup_restarts":
        warmup = LinearLR(optimizer, start_factor=0.01,
            end_factor=1.0, total_iters=warmup_steps,)

        cosine = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cfg.train.scheduler.T_0 * steps_per_epoch,
            T_mult=cfg.train.scheduler.T_mult,
            eta_min=cfg.train.scheduler.min_lr,
        )

        return SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_steps],
        )
    else:
        raise ValueError(
            f"Unsupported scheduler: {scheduler_type}")



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
    Ui:UI,
    dash:_TrainSession,
    device: torch.device) -> tuple[float, int]:
    
    '''
        Run one training epoch.
    
        Returns:
            (train_loss, updated_global_step)
    '''
 
    model.train()
    running_loss = 0.0
    num_samples = 0

    for i,(images,labels) in enumerate(dataloader):
         
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)# Gradient clipping to prevent exploding gradients

        dash.update_batch(loss=loss.item(), acc=(outputs.argmax(dim=1) == labels).float().mean().item())
        optimizer.step()
        scheduler.step(epoch + (i+1)/len(dataloader))  # type: ignore[arg-type]
        num_samples += batch_size
        global_step += 1

        #tensorboard loggings
        writer.add_scalar("Loss/train", loss.item(), global_step)
        writer.add_scalar("LearningRate", optimizer.param_groups[0]['lr'], global_step)
        
        #cli visual


        #pbar.set_postfix({
        #    "loss": f"{loss.item():.4f}",
        #    "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
        #})


    train_loss = running_loss / num_samples
    return train_loss, global_step

def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    metrics: dict,
    epoch: int,
    num_epochs: int,
    Ui:UI,
    dash:_TrainSession,
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

        for images, labels in dataloader:
            dash.start_validation(val_steps=len(dataloader))

            images, labels = images.to(device), labels.to(device)

            
            outputs = model(images)
            loss = criterion(outputs, labels)
            batch_size = images.size(0)

            dash.update_val_batch(loss=loss.item(), acc=(outputs.argmax(dim=1) == labels).float().mean().item())


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
