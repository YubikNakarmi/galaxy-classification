import click
import torch
import pandas as pd
from torchvision import transforms
import mlflow
import random
from dotenv import load_dotenv
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from torch.utils.data import DataLoader
import os
from loguru import logger
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC
import datetime
from mlflow.models.signature import infer_signature

from src.dataset.utils import generate_report
from src.dataset.utils import RandomCenterCrop
from src.utils.azure_utils import AzureBlobStorage
from src.models.model import GalaxyClassifier_ResNet
from src.utils.yaml_utils import load_yaml
from src.utils.logging_config import setup_ml_logging
from src.dataset.dataset import GalaxyDataset
from src.training import checkpoint, train_epoch,validate_epoch
from src.config import load_and_validate_config
from src.training import CheckpointManager,validate_setup


root = Path(__file__).parent.parent
load_dotenv(root / ".env")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI","http://localhost:5000")

log_config = load_yaml(root / 'configs' / 'core.yaml')
setup_ml_logging(log_config)

core_cfg = load_yaml(root / 'configs' / 'core.yaml')

file_config = root / 'configs' / 'ml.yaml'
ml_yaml = load_yaml(file_config)
cfg = load_and_validate_config(file_config)

#used throughout all loggings runs for local saves, cloud saves use mlflow run_id
experiment_name = cfg.active_variant + "_" + str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))  
dev = cfg.env.device
active_variant = cfg.active_variant



def make_metrics(num_classes:int, device: torch.device) -> dict:
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
# Hardcoded as there will be less editing of these transforms, but can be moved to config if needed
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
    transforms.CenterCrop(207),  
    transforms.Resize(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1376, 0.1276, 0.1187],
                         std= [0.1728, 0.1586, 0.1488])    
])

def register_model(model,val_loader:DataLoader):
    sample_images, _ = next(iter(val_loader))
    sample_images = sample_images.to(dev)
    
    # getting model signatures and sample for pt2 serialization
    with torch.no_grad():
        preds_np = torch.softmax(model(sample_images), dim=1).cpu().numpy()
 
    signature = infer_signature(sample_images.cpu().numpy(), preds_np)
 
    mlflow.pytorch.log_model(
        model,
        serialization_format="pt2", #most robust format for torchscript and future compatibility
        input_example=sample_images.cpu(),
        name="model",
        registered_model_name=cfg.variant.mlflow.model_name,
        signature=signature,
    )
    logger.info("Model logged to MLflow registry.")
 
def upload_artifact(blob: AzureBlobStorage, run_id: str)-> None:
    log_path = root / core_cfg["logging"]["ml"]["file"]
 
    url = blob.upload_blob(blob_name=f"{run_id}/logs/training.log", data=log_path)
    mlflow.set_tag("log_url", url)
 
    try:
        report_path = cfg.storage.local.logs.reports_dir / f"{experiment_name}.md"
        url = blob.upload_blob(blob_name=f"{run_id}/reports/report.md", data=report_path)
        mlflow.set_tag("report_url", url)

    except Exception as e:
        logger.error(f"Could not upload report: {e}")
 
    try:
        tb_run_dir = cfg.storage.local.logs.tensorboard_dir / experiment_name
        for file in tb_run_dir.iterdir():
            if file.is_file():
                url = blob.upload_blob(blob_name=f"{run_id}/tensorboard/{file.name}", data=file)
                mlflow.set_tag(f"tensorboard_{file.name}", url)

    except Exception as e:
        logger.error(f"Could not upload tensorboard logs: {e}")


def train_default():

    logger.info(f"Starting training for variant: {cfg.variant.name} with resume_enabled={cfg.train.checkpoints.resume_from_checkpoint}")
    random.seed(cfg.env.seed)
    torch.manual_seed(cfg.env.seed)
    torch.cuda.manual_seed_all(cfg.env.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    
    train_dataset = GalaxyDataset(labels_df=pd.read_csv(cfg.data.train_path), img_dir=cfg.data.img_dir, transform=raw_transforms)
    val_dataset = GalaxyDataset(labels_df=pd.read_csv(cfg.data.val_path), img_dir=cfg.data.img_dir, transform=val_transforms)

    train_sampler, train_class_weights = train_dataset.make_weighted_sampler()

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=False,sampler=train_sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.train.batch_size, shuffle=False)
    
    #load models and optimizers
    assert cfg.variant.num_classes is not None, \
    f"variant '{cfg.active_variant}' has num_classes=null, cannot use classification model"

    model = GalaxyClassifier_ResNet(num_classes=cfg.variant.num_classes)
    model.to(dev)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.learning_rate) #weight decay

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cfg.train.scheduler.warmup_epochs,eta_min =
                                                                      cfg.train.scheduler.min_lr 
                                                                     ,T_mult=2)
    
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(train_class_weights).to(dev))
    metrics   = make_metrics(cfg.variant.num_classes, torch.device(dev))
    writer    = SummaryWriter(log_dir=cfg.storage.local.logs.tensorboard_dir / experiment_name)
    
    az_checkpint_dir = cfg.train.checkpoints.checkpoint_restore_path

    chkpt = CheckpointManager(cfg=cfg,checkpoint_dir_local=cfg.train.checkpoints.local_checkpoint_path,
                                   checkpoint_dir_azure=az_checkpint_dir,
                                   run_id=cfg.train.checkpoints.run_id)
    
    # checkpoint manager already handles the logic for getting config if you want to resume or not
    chkpt_state = chkpt.load_checkpoint(model=model, optimizer=optimizer, scheduler=scheduler, device=dev)
    
    best_val_loss = chkpt_state.best_val_loss

    cfg_snapshot = {
        "variant":     cfg.active_variant,
        "num_classes": cfg.variant.num_classes,
        "lr":          cfg.train.learning_rate,
        "batch_size":  cfg.train.batch_size,
    }# to log in mlfow

    mlflow.enable_system_metrics_logging()

    with mlflow.start_run(run_name=experiment_name,run_id=chkpt_state.resume_run_id) as run:
        run_id = run.info.run_id
        chkpt.run_id = run_id #resuign same functino to update the run_id incase of new run
        logger.info(f"MLflow run: {run_id}")
 
        mlflow.log_params({
            "epochs":          cfg.train.num_epochs,
            "batch_size":      cfg.train.batch_size,
            "lr":              cfg.train.learning_rate,
            "weight_decay":    cfg.train.weight_decay,
            "grad_clip":       cfg.train.grad_clip_max_norm,
            "label_smoothing": cfg.train.label_smoothing,
            "variant":         cfg.active_variant,
            "num_classes":     cfg.variant.num_classes,
            "optimizer":       cfg.train.optimizer,
            "scheduler":       cfg.train.scheduler.type,
            "T_0":             cfg.train.scheduler.warmup_epochs,
            "T_mult":          2,
            "seed":            cfg.env.seed,
        })

        global_step = 0

        for ep in range(chkpt_state.start_epoch, cfg.train.num_epochs):
            train_loss, global_step = train_epoch(
                model, train_dataloader, optimizer, scheduler, criterion,
                writer, ep, cfg.train.num_epochs, global_step,
                cfg.train.grad_clip_max_norm, torch.device(dev),
            )
            
            val_loss, val_metrics = validate_epoch(
                model, val_dataloader, criterion, metrics,
                ep, cfg.train.num_epochs, torch.device(dev),
            )

            mlflow.log_metrics({
                "train_loss":          train_loss,
                "val_loss":            val_loss,
                "val_acc":             val_metrics["val_acc"],
                "val_f1_micro":        val_metrics["val_f1_micro"],
                "val_f1_macro":        val_metrics["val_f1_macro"],
                "val_precision_micro": val_metrics["val_precision_micro"],
                "val_precision_macro": val_metrics["val_precision_macro"],
                "val_recall_micro":    val_metrics["val_recall_micro"],
                "val_recall_macro":    val_metrics["val_recall_macro"],
                "val_roc_auc":         val_metrics["val_roc_auc"],
                "lr":                  optimizer.param_groups[0]["lr"],
            }, step=global_step)

            # Log classwise metrics to MLflow
            for i, (f1, prec, rec) in enumerate(zip(
                val_metrics["val_f1_classwise"],
                val_metrics["val_precision_classwise"],
                val_metrics["val_recall_classwise"],
            )):
                mlflow.log_metrics({
                    f"val_f1_class_{i}":        f1,
                    f"val_precision_class_{i}": prec,
                    f"val_recall_class_{i}":    rec,
                }, step=global_step)

            # handle tensorbaord checkpointing
            writer.add_scalar("Loss/val",     val_loss,                    ep)
            writer.add_scalar("Accuracy/val", val_metrics["val_acc"],      ep)
            writer.add_scalar("F1/val_macro", val_metrics["val_f1_macro"], ep)
 
            logger.info(
                f"Epoch {ep+1}/{cfg.train.num_epochs} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"val_acc={val_metrics['val_acc']:.4f} | val_f1_macro={val_metrics['val_f1_macro']:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                chkpt.save(model, optimizer, scheduler, ep, val_loss, val_metrics["val_f1_macro"], cfg_snapshot)
                ckpt_url =chkpt.upload_checkpoint_to_azure()#upload immediately after improvement for better restore
                mlflow.set_tag("checkpoint_url", ckpt_url)
                mlflow.set_tag("checkpoint_blob_path", f"checkpoints/{run_id}/best.pt")
        # after training   
        # 
        with AzureBlobStorage() as blob:
            blob.create_container(name=cfg.storage.cloud.experiments.container)   
            blob.create_container()     
      
     
        best_ckpt = torch.load(chkpt.checkpoint_dir, map_location=dev)
        model.load_state_dict(best_ckpt["model_state_dict"])
        logger.info(f"Loaded best weights from epoch {best_ckpt['epoch']+1}")
        
        register_model(model, val_dataloader)
        generate_report(run_id=run_id, output_path=cfg.storage.local.logs.reports_dir / f"{experiment_name}.md")
        upload_artifact(blob, run_id)

        writer.close()
        logger.success("Training complete.")

# CLI entry
@click.command()
@click.option("--variant", type=str, default="Default", help="Training variant to use")
def main(variant):

    if not validate_setup(required_paths=[cfg.data.img_dir,cfg.data.train_path,cfg.data.val_path], 
               mlflow_tracking_uri=MLFLOW_TRACKING_URI,
               
            create_dirs =   [cfg.storage.local.checkpoints.dir,
            cfg.storage.local.logs.reports_dir,
            cfg.storage.local.logs.tensorboard_dir]):
        
        logger.error("Setup validation failed. Exiting.")
        return
        
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if variant == "Default":
        mlflow.set_experiment(cfg.variant.experiment.root)
        train_default()
    
 
if __name__ == "__main__":
    main()
