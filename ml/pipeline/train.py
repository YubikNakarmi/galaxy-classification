from __future__ import annotations

import datetime
import random
from pathlib import Path

import click

from ml.pipeline import pipeline
from src.cli.style import UI








def train_default(cfg,ui:UI):# no console type hint for fast CLI
    import mlflow
    import pandas as pd
    import torch
    from loguru import logger
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter

    from src.dataset.dataset import GalaxyDataset
    from src.dataset.utils import generate_report
    from src.models.model import GalaxyClassifier_ResNet
    from src.training import (
        CheckpointManager,
        make_metrics,
        raw_transforms,
        register_model,
        train_epoch,
        upload_artifact,
        validate_epoch,
        val_transforms,
    )
    from src.utils.azure_utils import AzureBlobStorage

    #configs
    #used throughout all loggings runs for local saves, cloud saves use mlflow run_id
    experiment_name = cfg.active_variant + "_" + str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))  
    # normalize device to a torch.device once
    dev = torch.device(cfg.env.device)


    #trainin start

    logger.info(f"Starting training for variant: {cfg.variant.name} with resume_enabled={cfg.train.checkpoints.resume_from_checkpoint}")
    ui.info(f"Starting training for variant: {cfg.variant.name} with resume_enabled={cfg.train.checkpoints.resume_from_checkpoint}")

    logger.info(
        "Training configuration | device={} | seed={} | batch_size={} | num_epochs={} | train_path={} | val_path={}",
        cfg.env.device,
        cfg.env.seed,
        cfg.train.batch_size,
        cfg.train.num_epochs,
        cfg.data.train_path,
        cfg.data.val_path,
    )
    ui.debug(
        f"Training configuration | device={ cfg.env.device} | seed={cfg.env.seed} | \
        batch_size={cfg.train.batch_size} | num_epochs={cfg.train.num_epochs} |\
          train_path={cfg.data.train_path} | val_path={cfg.data.val_path}",
    )
      
    random.seed(cfg.env.seed)
    torch.manual_seed(cfg.env.seed)
    torch.cuda.manual_seed_all(cfg.env.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    

    
    train_dataset = GalaxyDataset(labels_df=pd.read_csv(cfg.data.train_path), img_dir=cfg.data.img_dir, transform=raw_transforms)
    val_dataset = GalaxyDataset(labels_df=pd.read_csv(cfg.data.val_path), img_dir=cfg.data.img_dir, transform=val_transforms)
    logger.info(
        "Loaded datasets | train_samples={} | val_samples={} | image_dir={}",
        len(train_dataset),
        len(val_dataset),
        cfg.data.img_dir,
    )

    ui.debug(
        f"Loaded datasets | train_samples={len(train_dataset)} | val_samples={len(val_dataset)} | image_dir={cfg.data.img_dir}",
    )

    train_sampler, train_class_weights = train_dataset.make_weighted_sampler()
    logger.info("Weighted sampler prepared | class_weights={}", train_class_weights)
    ui.debug(f"Weighted sampler prepared | class_weights={train_class_weights}")

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=False,sampler=train_sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.train.batch_size, shuffle=False)
    
    #load models and optimizers
    assert cfg.variant.num_classes is not None, \
    f"variant '{cfg.active_variant}' has num_classes=null, cannot use classification model"

    model = GalaxyClassifier_ResNet(num_classes=cfg.variant.num_classes)
    if dev != torch.device("cuda"):
        logger.warning("CUDA device not available, training will be performed on CPU. This may be slow.")
        ui.warning("CUDA device not available, training will be performed on CPU. This may be slow.")

    model.to(dev)

    

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.learning_rate) #weight decay

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cfg.train.scheduler.warmup_epochs,eta_min =
                                                                      cfg.train.scheduler.min_lr 
                                                                     ,T_mult=2)
    
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(train_class_weights).to(dev))
    metrics   = make_metrics(cfg.variant.num_classes, dev)
    writer    = SummaryWriter(log_dir=cfg.storage.local.logs.tensorboard_dir / experiment_name)

    ui.debug(f"TensorBoard writer initialized | log_dir={cfg.storage.local.logs.tensorboard_dir / experiment_name}")

    az_checkpoint_dir = cfg.train.checkpoints.checkpoint_restore_path
    chkpt_local_path = Path(cfg.train.checkpoints.local_checkpoint_path) / experiment_name   

    chkpt = CheckpointManager(cfg=cfg,checkpoint_dir_local=chkpt_local_path,
                                   checkpoint_dir_azure=az_checkpoint_dir,
                                   run_id=cfg.train.checkpoints.run_id)
    logger.info(
        "Checkpoint manager initialized | local_dir={} | azure_dir={} | resume_enabled={}",
        chkpt_local_path,
        az_checkpoint_dir,
        cfg.train.checkpoints.resume_from_checkpoint,
    )

    ui.info(
        f"Checkpoint manager initialized | local_dir={chkpt_local_path} | \
        azure_dir={az_checkpoint_dir} | resume_enabled={cfg.train.checkpoints.resume_from_checkpoint}",)
    
    # checkpoint manager already handles the logic for getting config if you want to resume or not
    chkpt_state = chkpt.load_checkpoint(model=model, optimizer=optimizer, scheduler=scheduler, device=dev,save_filename="tmp.pt")
    
    best_val_loss = chkpt_state.best_val_loss

    cfg_snapshot = {
        "variant":     cfg.active_variant,
        "num_classes": cfg.variant.num_classes,
        "lr":          cfg.train.learning_rate,
        "batch_size":  cfg.train.batch_size,
    }# to log in mlfow

    mlflow.enable_system_metrics_logging()
    ui.trace(f"MLflow system metrics logging enabled | tracking_uri={mlflow.get_tracking_uri()}")

    with mlflow.start_run(run_name=experiment_name,run_id=chkpt_state.resume_run_id) as run:
        run_id = run.info.run_id
        chkpt.run_id = run_id #resuign same functino to update the run_id incase of new run
        logger.info(
            "MLflow run started | run_id={} | resume_run_id={} | start_epoch={} | best_val_loss={}",
            run_id,
            chkpt_state.resume_run_id,
            chkpt_state.start_epoch,
            chkpt_state.best_val_loss,
        )

        ui.info(
            f"MLflow run started | run_id={run_id} | resume_run_id={chkpt_state.resume_run_id} |\
              start_epoch={chkpt_state.start_epoch} | best_val_loss={chkpt_state.best_val_loss}",
        )

        if chkpt_state.resume_run_id is None:
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
        else:
            logger.info("Resumed MLflow run detected; skipping duplicate parameter logging.")
            ui.info("Resumed MLflow run detected; skipping duplicate parameter logging.")

        global_step = 0

        for ep in range(chkpt_state.start_epoch, cfg.train.num_epochs):
            # need to use rich for a bar during training
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

            ui.info(
                f"Epoch {ep+1}/{cfg.train.num_epochs} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"val_acc={val_metrics['val_acc']:.4f} | val_f1_macro={val_metrics['val_f1_macro']:.4f}"
            )

           

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                logger.info("New best validation loss {:.4f} at epoch {}", val_loss, ep + 1)
                ui.info(f"New best validation loss {val_loss:.4f} at epoch {ep + 1}")

                chkpt.save(model, optimizer, scheduler, ep, val_loss, val_metrics["val_f1_macro"], cfg_snapshot, filename="best.pt")
                ckpt_url =chkpt.upload_checkpoint_to_azure(filename="best.pt")#upload immediately after improvement for better restore
                mlflow.set_tag("checkpoint_url", ckpt_url)
                mlflow.set_tag("checkpoint_blob_path", f"checkpoints/{run_id}/best.pt")
                logger.info("Best checkpoint synced to Azure | url={}", ckpt_url)

                ui.info(f"Best checkpoint synced to Azure | url={ckpt_url}")
        # after training   
        # 
        with AzureBlobStorage() as blob:
            # ensure experiment container exists
            logger.info("Ensuring Azure artifact container exists: {}", cfg.storage.cloud.experiments.container)
            ui.debug(f"Ensuring Azure artifact container exists: {cfg.storage.cloud.experiments.container}")
            blob.create_container(name=cfg.storage.cloud.experiments.container)

            # load best checkpoint (support either file path or directory containing best.pt)
            best_candidate = Path(chkpt.checkpoint_dir)
            if best_candidate.is_file():
                best_ckpt_path = best_candidate
            else:
                best_ckpt_path = best_candidate / "best.pt"

            best_ckpt = torch.load(best_ckpt_path, map_location=dev)
            model.load_state_dict(best_ckpt["model_state_dict"])

            logger.info(
                "Loaded best weights | epoch={} | checkpoint_path={}",
                best_ckpt["epoch"] + 1,
                best_ckpt_path,
            )
            ui.info(
                f"Loaded best weights | epoch={best_ckpt['epoch'] + 1} | checkpoint_path={best_ckpt_path}",
            )

            logger.info("Registering model to mlflow")
            ui.debug("Registering model to mlflow")
            register_model(model, val_dataloader)

            logger.info("Generating training report")
            ui.debug("Generating training report")
            generate_report(run_id=run_id, output_path=cfg.storage.local.logs.reports_dir / f"{experiment_name}.md")

            logger.info("Uploading artifacts to Azure Blob Storage")
            ui.debug("Uploading artifacts to Azure Blob Storage")
            upload_artifact(blob, run_id,experiment_name=experiment_name)

        writer.close()
        logger.success("Training complete.")
           
    mlflow.end_run()
    return run_id


def train_hubble():
    pass

def train_dirilchled():
    pass
# CLI entry
@pipeline.command()
@click.option("--variant", type=str, default="Default", help="Training variant to use")
@click.option("--evaluate", is_flag=True, default=False, help="Evaluate the model after training")
@click.pass_context

def train(ctx, variant:str, evaluate:bool):

    import mlflow
    from loguru import logger

    from src.training import validate_setup
    from src.utils.bootstrap import load_context
    from ml.pipeline.eval import evaluate as eval

    ui = ctx.find_object(UI) or UI(verbose=0)  # Retrieve the UI object from the context or create a new one
    

    app_ctx = ctx.obj or load_context(service="ml") #retrive from top level of click cmd in __init__py, fallback to load_context if not present
    cfg = app_ctx.cfg
    MLFLOW_TRACKING_URI = app_ctx.mlflow_tracking_uri


    if not validate_setup(required_paths=[cfg.data.img_dir,cfg.data.train_path,cfg.data.val_path], 
               mlflow_tracking_uri=MLFLOW_TRACKING_URI,
               
            create_dirs =   [cfg.storage.local.checkpoints.dir,
            cfg.storage.local.logs.reports_dir,
            cfg.storage.local.logs.tensorboard_dir]):
        
        logger.error("Setup validation failed. Exiting.")
        ui.error("setup validation failed. Please check the logs for details.")
        return
        
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if variant == "Default":
        mlflow.set_experiment(cfg.variant.experiment.root)
        run_id = train_default(cfg,ui)
        ui.success("Training complete." ,style = "bold spring_green2")

        if evaluate:
            ui.info("Starting evaluation of the trained model...")
            eval(ctx, run_id=run_id, variant=variant, model_version=None,
                  min_improvement=0.005, metric_name="f1_macro", alias="Champion", 
                  model_source="mlflow")

    
 
if __name__ == "__main__":
    train()
