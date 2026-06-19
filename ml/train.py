
#Need to refactor some code in order be make it more clean with other models. Like refactoring Azure uploads into one function. 
import click
import torch
import pandas as pd
from src.dataset.dataset import GalaxyDataset
from torchvision import transforms
import mlflow
import random
from dotenv import load_dotenv
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from torch.utils.data import DataLoader
import os
from src.models.model import GalaxyClassifier_ResNet
from src.utils.yaml_utils import load_yaml
from src.utils.logging_config import setup_ml_logging
from loguru import logger
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC
from tqdm import tqdm
from src.utils.azure_utils import AzureBlobStorage
import datetime
from mlflow.models.signature import infer_signature
from src.dataset.utils import generate_report
from src.dataset.utils import RandomCenterCrop

root = Path(__file__).parent.parent
load_dotenv(root / ".env")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI","http://localhost:5000")


log_config = load_yaml(root / 'configs' / 'core.yaml')
setup_ml_logging(log_config)


file_config = root / 'configs' / 'ml.yaml'
ml_yaml = load_yaml(file_config)
dev = ml_yaml["env"]["device"]

device = torch.device(dev)
seed = ml_yaml["env"]["seed"]

checkpoint_resume= ml_yaml["train"]["checkpoints"]["resume_from_checkpoint"]
checkpoint_dir_blob = ml_yaml["train"]["checkpoints"]["checkpoint_restore_path"]
checkpoint_dir_local = root / ml_yaml["train"]["checkpoints"]["local_checkpoint_path"]
checkpoint_dir_local.mkdir(parents=True, exist_ok=True).touch("resume_best.pt")
checkpoint_hard = ml_yaml["train"]["checkpoints"]["checkpoint_resume_hard"] #if true, means that users exits if there is no checkpoint to resume from. If false, it will just start fresh training.

epoch = int(ml_yaml["train"]["num_epochs"])
image_dir = root / ml_yaml["data"]["img_dir"]
batch_size = int(ml_yaml["train"]["batch_size"])

train_path = root /ml_yaml["data"]["train_path"]
val_path = root / ml_yaml["data"]["val_path"] 
active = str(ml_yaml["active_variant"])

experiment_root = ml_yaml["variants"][ml_yaml["active_variant"]]["experiment"]["root"] #top level experiment name
experiment_name = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) + "_" + active #used throughout all loggings runs
checkpoint_dir = root / ml_yaml["storage"]["local"]["checkpoints"]["dir"] / experiment_name
checkpoint_dir.mkdir(parents=True, exist_ok=True)
checkpoint_container_name = ml_yaml["storage"]["cloud"]["experiments"]["container"]

log_dir = root / log_config["logging"]["ml"]["file"]
reports_dir = root / ml_yaml["storage"]["local"]["logs"]["reports_dir"]

tensorboard_dir = root / ml_yaml["storage"]["local"]["logs"]["tensorboard_dir"]
tensorboard_path = tensorboard_dir / experiment_name

num_class = ml_yaml["variants"][active]["num_classes"]
lr = ml_yaml["train"]["learning_rate"]
eta_min = ml_yaml["train"]["scheduler"]["min_lr"]
warmup_epochs = ml_yaml["train"]["scheduler"]["warmup_epochs"]
optimizer = ml_yaml["train"]["optimizer"]
grad_clip = ml_yaml["train"]["grad_clip_max_norm"]

model_registry_name = ml_yaml["variants"][active]["mlflow"]["model_name"]


writer = SummaryWriter(log_dir=tensorboard_path)

best_val_loss = float('inf')
best_val_model = None
 

raw_transforms = transforms.Compose([
    RandomCenterCrop(170, 240), 
    transforms.Resize(160),                            
    transforms.RandomCrop(128),                        
    transforms.RandomRotation(degrees = [0, 360]),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.5, saturation=0.4, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1376, 0.1276, 0.1187],
                         std= [0.1728, 0.1586, 0.1488])    
])

val_transforms = transforms.Compose([
    RandomCenterCrop(170, 240),
    transforms.Resize(128),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.1376, 0.1276, 0.1187],
        std=[0.1728, 0.1586, 0.1488]
    )
])


def validate_setup() -> bool:
    missing = []
    for p in [train_path, val_path, image_dir, checkpoint_dir, log_dir, tensorboard_dir,checkpoint_dir_local, reports_dir]:
        if not Path(p).exists():
            missing.append(str(p))

    if missing:
        logger.exception(f"Missing paths: {', '.join(missing)}")
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
    logger.info("All setup checks passed.")
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


def train_default(tensorboard:bool=True,report:bool=True,checkpoint_resume_hard:bool=False,resume_from_checkpoint=checkpoint_resume):


    logger.info(f"Starting training with variant '{active}'")
    logger.info(f"Experiment name: {experiment_name}")

    if resume_from_checkpoint:
        logger.warning("Checkpoint resume enabled.")

    logger.warning(f"Checkpoint resume hard mode: {checkpoint_resume_hard}")

    logger.info(f"Resume from checkpoint: {resume_from_checkpoint}")

    #setting seed 
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    
    metrics = make_metrics(num_class, device) #cleaner version of metrics setup

    
    #load data and samplers
    train_label_df = pd.read_csv(train_path)
    val_label_df = pd.read_csv(val_path)

    train_dataset = GalaxyDataset(labels_df=train_label_df, img_dir=image_dir, transform=raw_transforms)
    val_dataset = GalaxyDataset(labels_df=val_label_df, img_dir=image_dir, transform=val_transforms)

    train_sampler, train_class_weights = train_dataset.make_weighted_sampler()

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,sampler=train_sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    #load models and optimizers
    model = GalaxyClassifier_ResNet(num_classes=num_class)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr) #weight decay
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=warmup_epochs,eta_min = eta_min ,T_mult=2)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(train_class_weights).to(device))

    best_val_loss = float("inf")
    best_checkpoint_path = checkpoint_dir / "best.pt"

    global_step = 0 #for scheeduler 

    # For resuming training from checkpoint if available
    if resume_from_checkpoint and Path(checkpoint_dir_local / "resume_best.pt").exists():
        blob_helper = AzureBlobStorage(container_name = checkpoint_container_name)
        blob_helper.load_blob(blob_name=checkpoint_dir_blob, local_path=checkpoint_dir_local / "resume_best.pt")
        
        ckpt = torch.load(checkpoint_dir_local / "resume_best.pt", map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt["val_loss"]
        logger.info(f"Resumed from epoch {start_epoch}")
    else:

        if checkpoint_resume_hard:#if true, means that users exits if there is no checkpoint to resume from. If false, it will just start fresh training.
            logger.error("Checkpoint resume requested but no checkpoint found. Exiting.")
            raise logger.exception("No checkpoint found for resuming training.")
            
        logger.info("No checkpoint found, starting fresh training.")
        start_epoch = 0
  
    resume_run_id = ckpt.get("run_id") if resume_from_checkpoint else None
    mlflow.enable_system_metrics_logging()
    
    with mlflow.start_run(run_id=resume_run_id, run_name=experiment_name) as run:
        run_id = run.info.run_id
        logger.info(f"MLflow run started: {run_id}")

        mlflow.log_params({
            "epochs": epoch,
            "batch_size": batch_size,
            "lr": lr,
            "grad_clip": grad_clip,
            "variant": active,
            "num_classes": num_class,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingWarmRestarts",
            "T_0": warmup_epochs,
            "T_mult": 2,
            "label_smoothing": 0.1,
            "batch_size": batch_size,
            "variant": active,
            "num_classes": num_class,
            "seed": seed
            
        })

        for ep in range(start_epoch, epoch):
            model.train()
            train_loss = 0.0

            pbar = tqdm(train_dataloader, desc=f"Epoch {ep+1}/{epoch} [train]")

            ''' Training'''
            for images, labels in pbar:

                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                train_loss += loss.item() * images.size(0)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)# Gradient clipping to prevent exploding gradients


                optimizer.step()
                scheduler.step(ep+ global_step/len(train_dataloader)) # Update scheduler with epoch and batch progress
                
                global_step += 1

                #tensorboard loggings
                writer.add_scalar("Loss/train", loss.item(), global_step)
                writer.add_scalar("LearningRate", optimizer.param_groups[0]['lr'], global_step)
                
                #cli visual
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
                })
 
            train_loss /= len(train_dataset)

            
            ''' Evaluation'''
            model.eval()
            val_loss = 0.0
            with torch.no_grad():

                for images, labels in tqdm(val_dataloader, desc=f"Epoch {ep+1}/{epoch} [val]"):
                    images, labels = images.to(device), labels.to(device)

                    
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


        ''' after training steps'''        

        logger.info("Training complete. Uploading checkpoint to Azure Blob...")
        blob_path = f"checkpoints/{run_id}/best.pt"

        blob_helper_checkpoints = AzureBlobStorage(container_name=checkpoint_container_name)

        #uploading best checkpoint to azure
        blob_helper_checkpoints.create_container() 

        with open(best_checkpoint_path, "rb") as f:
            blob_url = blob_helper_checkpoints.upload_blob(blob_name=blob_path, data=f)
            
        mlflow.set_tag("checkpoint_url", blob_url)
        mlflow.set_tag("checkpoint_blob_path", blob_path)



        # Load best weights before logging to registry 
        best_ckpt = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(best_ckpt["model_state_dict"])
        logger.info(f"Loaded best weights from epoch {best_ckpt['epoch']+1}")


        #infer signatures for mlflow registry 
        sample_images, _ = next(iter(val_dataloader))
        sample_images = sample_images.to(device)

        with torch.no_grad():
            sample_preds = model(sample_images)
            sample_preds_np = torch.softmax(sample_preds, dim=1).cpu().numpy()

        signature = infer_signature(
            sample_images.cpu().numpy(),
            sample_preds_np
        )
        

        #logging to mlflwo registry


        mlflow.pytorch.log_model(
            model,
            serialization_format="pt2", #most stable one
            input_example=sample_images.cpu(),
            name="model",
            registered_model_name=ml_yaml["variants"][active].get("mlflow", {}).get("model_name"),
            signature=signature,
        )
        logger.info("Model logged to MLflow registry.")

        generate_report(run_id=run_id, output_path=reports_dir / f"{experiment_name}.md")
        logger.info("report generated at {}")


          #uploading logs to azure
        blob_helper_logs = AzureBlobStorage(container_name = log_config["logging"]["cloud"]["logs"]["container"])
        log_blob_path = f"logs/{run_id}/training.log"
        blob_helper_logs.create_container()

        with open(str(log_dir), "rb") as f:
            blob_log_url= blob_helper_logs.upload_blob(blob_name=log_blob_path, data=f)
         
        mlflow.set_tag("log_url", blob_log_url)

        # with exceptions to run even if there are erros saving
        try:
            with open(reports_dir / f"{experiment_name}.md", "rb") as f:
                report_blob_path = f"reports/{run_id}/report.md"
                report_blob_url = blob_helper_logs.upload_blob(blob_name=report_blob_path, data=f) #same container as logs

        except Exception as e:
            logger.error(f"couldnt save report data {e}")

        mlflow.set_tag("report_url", report_blob_url)

        try:
            for path in Path(tensorboard_dir).iterdir():
                if path.is_dir() and path.name == experiment_name:
                    for file in path.iterdir():
                        if file.is_file():
                            tb_blob_path = f"tensorboard/{run_id}/{file.name}"
                            with open(file, "rb") as f:
                                tb_blob_url = blob_helper_logs.upload_blob(blob_name=tb_blob_path, data=f)
                            mlflow.set_tag(f"tensorboard_{file.name}", tb_blob_url)
                            
        except Exception as e:
            logger.error(f"Error uploading TensorBoard logs: {e}")

        logger.info(f"report generated at {reports_dir / f'{experiment_name}.md'}")
        writer.close()
        
        logger.info("Training Completed")

    






        
@click.command()
@click.option("--variant", type=str, default="default", help="Training variant to use")
@click.option("--checkpoint_resume_force", is_flag=True, default=checkpoint_hard,help="If set, the training will exit if no " \
"checkpoint is found to resume from. If not set, it will start fresh training.")
@click.option("--resume_checkpoint", is_flag=True, default=checkpoint_resume, help="Whether to attempt resuming from checkpoint if available.")

def main(variant, checkpoint_resume_force,resume_checkpoint):
    
    if validate_setup():
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        if variant == "default":
            mlflow.set_experiment(experiment_root)
            train_default(checkpoint_resume_hard=checkpoint_resume_force)


if __name__ == "__main__":
    main()