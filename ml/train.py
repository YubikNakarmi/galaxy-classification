import click
import torch
import pandas as pd
import mlflow
import random
from dotenv import load_dotenv
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from torch.utils.data import DataLoader
import os
from loguru import logger
import datetime

from src.dataset.utils import generate_report
from src.utils.azure_utils import AzureBlobStorage
from src.models.model import GalaxyClassifier_ResNet
from src.utils.yaml_utils import load_yaml
from src.utils.logging_config import setup_ml_logging
from src.dataset.dataset import GalaxyDataset
from src.training import train_epoch,validate_epoch
from src.config import load_and_validate_config
from src.training import CheckpointManager,validate_setup,upload_artifact,register_model,make_metrics, raw_transforms, val_transforms

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
# normalize device to a torch.device once
dev = torch.device(cfg.env.device)
active_variant = cfg.active_variant


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
    metrics   = make_metrics(cfg.variant.num_classes, dev)
    writer    = SummaryWriter(log_dir=cfg.storage.local.logs.tensorboard_dir / experiment_name)
    
    az_checkpoint_dir = cfg.train.checkpoints.checkpoint_restore_path
    chkpt_local_path = Path(cfg.train.checkpoints.local_checkpoint_path) / experiment_name   

    chkpt = CheckpointManager(cfg=cfg,checkpoint_dir_local=chkpt_local_path,
                                   checkpoint_dir_azure=az_checkpoint_dir,
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
                cfg.train.grad_clip_max_norm, dev,
            )
            
            val_loss, val_metrics = validate_epoch(
                model, val_dataloader, criterion, metrics,
                ep, cfg.train.num_epochs, dev,
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
                chkpt.save(model, optimizer, scheduler, ep, val_loss, val_metrics["val_f1_macro"], cfg_snapshot, filename="best.pt")
                ckpt_url =chkpt.upload_checkpoint_to_azure(filename="best.pt")#upload immediately after improvement for better restore
                mlflow.set_tag("checkpoint_url", ckpt_url)
                mlflow.set_tag("checkpoint_blob_path", f"checkpoints/{run_id}/best.pt")
        # after training   
        # 
        with AzureBlobStorage() as blob:
            # ensure experiment container exists
            blob.create_container(name=cfg.storage.cloud.experiments.container)

            # load best checkpoint (support either file path or directory containing best.pt)
            best_candidate = Path(chkpt.checkpoint_dir)
            if best_candidate.is_file():
                best_ckpt_path = best_candidate
            else:
                best_ckpt_path = best_candidate / "best.pt"

            best_ckpt = torch.load(best_ckpt_path, map_location=dev)
            model.load_state_dict(best_ckpt["model_state_dict"])
            logger.info(f"Loaded best weights from epoch {best_ckpt['epoch']+1}")
            
            register_model(model, val_dataloader)
            generate_report(run_id=run_id, output_path=cfg.storage.local.logs.reports_dir / f"{experiment_name}.md")
            upload_artifact(blob, run_id)

        writer.close()
        logger.success("Training complete.")


def train_hubble():
    pass

def train_dirilchled():
    pass
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
