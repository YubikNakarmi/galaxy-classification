import click
from functorch import dim
from matplotlib.pyplot import step
import torch
import pandas as pd
import src.dataset.dataset as dataset
import src.models.model as model
from src.dataset.dataset import GalaxyDataset
from torch.utils.data import DataLoader, dataset
from torchvision import transforms
import mlflow
import random
from dotenv import load_dotenv
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import os
from src.models.model import GalaxyClassifier_ResNet
from src.utils.yaml_utils import load_yaml
from src.utils.logging_config import setup_ml_logging
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix, AUROC
from tqdm import tqdm
from torch import nn
from src.utils.azure_utils import AzureBlobStorage
import datetime


root = Path(__file__).parent.parent
load_dotenv(root / ".env")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")


log_config = load_yaml(root / 'configs' / 'core.yaml')
setup_ml_logging(log_config)


file_config = root / 'configs' / 'ml.yaml'
ml_yaml = load_yaml(file_config)
dev = ml_yaml["env"]["device"]

device = torch.device(dev)

epoch = int(ml_yaml["train"]["num_epochs"])
image_dir = ml_yaml["data"]["img_dir"]
split_dir = ml_yaml["data"]["split_dir"]
batch_size = int(ml_yaml["train"]["batch_size"])

train_path = Path(split_dir) / "train_split.csv"
val_path = Path(split_dir) / "val_split.csv"

active = str(ml_yaml["active_variant"])

experiment_root = ml_yaml["variants"][ml_yaml["active_variant"]]["experiment"]["root"]
experiment_name = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
checkpoint_dir = root / ml_yaml["storage"]["local"]["checkpoints"]["dir"] / experiment_name
checkpoint_dir.mkdir(parents=True, exist_ok=True)


tensorboard_dir = root / ml_yaml["storage"]["local"]["tensorboard"]["dir"]

num_class = ml_yaml["variants"][active]["num_classes"]
lr = ml_yaml["train"]["learning_rate"]
grad_clip = ml_yaml["train"]["grad_clip"]
eta_min = ml_yaml["train"]["scheduler"]["eta_min"]
warmup_epochs = ml_yaml["train"]["scheduler"]["warmup_epochs"]
optimizer = ml_yaml["train"]["optimizer"]


writer = SummaryWriter(log_dir=tensorboard_dir / experiment_name)



best_val_loss = float('inf')
best_val_model = None


raw_transforms = transforms.Compose([
    transforms.CenterCrop(random.randint(170, 240)),  
    transforms.Resize(160),                            
    transforms.RandomCrop(128),                        
    transforms.RandomRotation(degrees = [90, 180]),
    transforms.RandomRotation(degrees = [0,270]),      
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1376, 0.1276, 0.1187],
                         std= [0.1728, 0.1586, 0.1488])    
])

val_transforms = transforms.Compose([
    transforms.CenterCrop(200),
    transforms.Resize(128),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.1376, 0.1276, 0.1187],
        std=[0.1728, 0.1586, 0.1488]
    )
])


def validate_setup() -> bool:
    missing = []
    for p in [train_path, val_path, Path(image_dir), Path(split_dir)]:
        if not Path(p).exists():
            missing.append(str(p))
    if missing:
        logger.error(f"Missing paths: {missing}")
        return False
 
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("Connection Test")
        with mlflow.start_run(run_name="setup_validation"):
            mlflow.log_param("validation", "success")
        logger.info("MLflow connection OK")
    except Exception as e:
        logger.error(f"MLflow not reachable: {e}")
        return False
 
    return True


def make_metrics(num_classes, device):
    return {
        "accuracy":           Accuracy(task="multiclass", num_classes=num_classes).to(device),
        "f1_micro":           F1Score(task="multiclass", average="micro", num_classes=num_classes).to(device),
        "f1_macro":           F1Score(task="multiclass", average="macro", num_classes=num_classes).to(device),
        "f1_classwise":       F1Score(task="multiclass", average="none", num_classes=num_classes).to(device),
        "precision_micro":    Precision(task="multiclass", average="micro", num_classes=num_classes).to(device),
        "precision_macro":    Precision(task="multiclass", average="macro", num_classes=num_classes).to(device),
        "precision_classwise":Precision(task="multiclass", average="none", num_classes=num_classes).to(device),
        "recall_micro":       Recall(task="multiclass", average="micro", num_classes=num_classes).to(device),
        "recall_macro":       Recall(task="multiclass", average="macro", num_classes=num_classes).to(device),
        "recall_classwise":   Recall(task="multiclass", average="none", num_classes=num_classes).to(device),
        "roc_auc":            AUROC(task="multiclass", num_classes=num_classes).to(device),
    }

def reset_metrics(metrics: dict): #reset all metrics at the end of each epoch
    for m in metrics.values():
        m.reset()


def train_default():
    
    metrics = make_metrics(num_class, device) #cleaner version of metrics setup

    mlflow.log_params({
    "epochs": epoch,
    "batch_size": batch_size,
    "variant": active,
    "num_classes": num_class,
})


    
    #load data and samplers
    train_label_df = pd.read_csv(train_path)
    val_label_df = pd.read_csv(val_path)

    train_dataset = GalaxyDataset(labels_df=train_label_df, img_dir=image_dir, transform=raw_transforms)
    val_dataset = GalaxyDataset(labels_df=val_label_df, img_dir=image_dir, transform=raw_transforms)

    train_sampler, train_class_weights = train_dataset.make_weighted_sampler()

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,sampler=train_sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    #load models and optimizers
    model = GalaxyClassifier_ResNet(num_classes=5)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=warmup_epochs,eta_min = eta_min ,T_mult=2)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(train_class_weights).to(device))

    best_val_loss = float("inf")
    best_checkpoint_path = checkpoint_dir / "best.pt"
    global_step = 0

    global_step = 0 #for scheeduler 
    
    with mlflow.start_run(run_name=experiment_name) as run:

        run_id = run.info.run_id
        logger.info(f"MLflow run started: {run_id}")
        accuracy = Accuracy(task = "multiclass", num_classes=5).to(device)

        mlflow.log_params({
            "epochs": epoch,
            "batch_size": batch_size,
            "lr": lr,
            "grad_clip": grad_clip,
            "variant": active,
            "num_classes": num_class,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingWarmRestarts",
            "T_0": 5,
            "T_mult": 2,
            "label_smoothing": 0.1,
        })

        for ep in range(epoch):
            model.train()
            train_loss = 0.0

            pbar = tqdm(train_dataloader, desc=f"Epoch {ep+1}/{epoch} [train]")

            for images, labels in pbar:

                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)# Gradient clipping to prevent exploding gradients


                optimizer.step()
                scheduler.step(ep+ global_step/len(train_dataloader)) # Update scheduler with epoch and batch progress
                
                global_step += 1

                writer.add_scalar("Loss/train", loss.item(), global_step)
                writer.add_scalar("LearningRate", optimizer.param_groups[0]['lr'], global_step)
                
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
                })
 
            train_loss /= len(train_dataset)

            

            model.eval()
            val_loss = 0.0

            for images, labels in pbar(val_dataloader, desc=f"Epoch {ep+1}/{num_epochs} [val]"):
                images, labels = images.to(device), labels.to(device)
                with torch.no_grad():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * images.size(0)

                    preds = torch.softmax(outputs, dim=1)

                    #all metrics
                    metrics["accuracy"].update(preds, labels)
                    metrics["f1_micro"].update(preds, labels)
                    metrics["f1_macro"].update(preds, labels)
                    metrics["f1_classwise"].update(preds, labels)
                    metrics["precision_micro"].update(preds, labels)
                    metrics["precision_macro"].update(preds, labels)
                    metrics["precision_classwise"].update(preds, labels)
                    metrics["recall_micro"].update(preds, labels)
                    metrics["recall_macro"].update(preds, labels)
                    metrics["recall_classwise"].update(preds, labels)
                    metrics["roc_auc"].update(preds, labels)

            val_loss /= len(val_dataset)



            val_acc = metrics["accuracy"].compute().item()
            val_f1_micro = metrics["f1_micro"].compute().item()
            val_f1_macro = metrics["f1_macro"].compute().item()
            val_f1_classwise = metrics["f1_classwise"].compute().tolist()
            val_prec_micro = metrics["precision_micro"].compute().item()
            val_prec_macro = metrics["precision_macro"].compute().item()
            val_prec_classwise = metrics["precision_classwise"].compute().tolist()
            val_recall_micro = metrics["recall_micro"].compute().item()
            val_recall_macro = metrics["recall_macro"].compute().item()
            val_recall_classwise = metrics["recall_classwise"].compute().tolist()
            val_roc_auc = metrics["roc_auc"].compute().item()
 
            reset_metrics(metrics)

            #per epoch mlflow log
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1_micro": val_f1_micro,
                "val_f1_macro": val_f1_macro,
                "val_precision_micro": val_prec_micro,
                "val_precision_macro": val_prec_macro,
                "val_recall_micro": val_recall_micro,
                "val_recall_macro": val_recall_macro,
                "val_roc_auc": val_roc_auc,
                "lr": optimizer.param_groups[0]["lr"],
            }, step=ep)

            #class wise metrics logging
            for i, (f1, prec, rec) in enumerate(zip(val_f1_classwise, val_prec_classwise, val_recall_classwise)):
                mlflow.log_metrics({
                    f"val_f1_class_{i}": f1,
                    f"val_precision_class_{i}": prec,
                    f"val_recall_class_{i}": rec,
                }, step=ep)
    
           # TensorBoard loggings
            writer.add_scalar("Loss/val", val_loss, ep)
            writer.add_scalar("Accuracy/val", val_acc, ep)
            writer.add_scalar("F1/val_macro", val_f1_macro, ep)

            logger.info(
                f"Epoch {ep+1}/{epoch} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"val_acc={val_acc:.4f} | val_f1_macro={val_f1_macro:.4f}"
            )

            logger.info(
                f"Epoch {ep+1}/{epoch} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"val_acc={val_acc:.4f} | val_f1_macro={val_f1_macro:.4f}"
            )

            # checkpoints with best val models

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    "epoch": ep,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_loss": val_loss,
                    "val_f1_macro": val_f1_macro,
                    "run_id": run_id,
                    "config": {
                        "variant": active,
                        "num_classes": num_class,
                        "lr": lr,
                        "batch_size": batch_size,
                    }
                }, best_checkpoint_path)
                logger.info(f"New best checkpoint saved (val_loss={val_loss:.4f})")

        logger.info("Training complete. Uploading checkpoint to Azure Blob...")
        blob_path = f"checkpoints/{run_id}/best.pt"

        blob_helper = AzureBlobStorage(container_name=ml_yaml["storage"]["azure"]["container_name"])
        blob_url = upload_to_blob(best_checkpoint_path, blob_path)
        mlflow.set_tag("checkpoint_url", blob_url)
        mlflow.set_tag("checkpoint_blob_path", blob_path)

        # Load best weights before logging to registry
        best_ckpt = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(best_ckpt["model_state_dict"])
        logger.info(f"Loaded best weights from epoch {best_ckpt['epoch']+1}")

        mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            registered_model_name=ml_yaml["variants"][active].get("mlflow", {}).get("model_name"),
        )
        logger.info("Model logged to MLflow registry.")

        writer.close()
        logger.info("Training Completed")

    


        
@click.command()
@click.option("--variant", type=str, default="default", help="Training variant to use")


def main(variant):
    validation()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    if variant == "default":
        mlflow.set_experiment(experiment_name)
        train_default()
    



if __name__ == "__main__":
    main()
