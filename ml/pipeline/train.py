from __future__ import annotations

import datetime
import random
from pathlib import Path
from typing import Any, Optional, Literal
import click
from ml.pipeline import pipeline
from src.cli.style import UI

def cycle_ends(t0: int, t_mult: int, max_epoch: int) -> set[int]:
    """Epoch counts (1-indexed) at which each cosine cycle finishes."""
    
    ends, length, total = set(), t0, 0
    while total + length <= max_epoch:
        total += length
        ends.add(total)
        length *= t_mult
    return ends


def train_default(cfg, ui:UI, resume:Optional[bool]=None,
                  resume_id:Optional[str]=None,
                  chkpt_type:Literal["Best","Latest"]="Best") -> str:
    # no console type hint for fast CLI
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
    
    
    CHKPT_TYPE = {"Best": "best.pt",
                  "Latest": "latest.pt"}
    #configs
    #used throughout all loggings runs for local saves, cloud saves use mlflow run_id
    experiment_name = cfg.active_variant + "_" + str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))  
    # normalize device to a torch.device once
    dev = torch.device(cfg.env.device)
    
    LATEST_UPLOAD_EPOCHS = cfg.train.checkpoint.save_freq_epoch


    #trainin start

    ui.info(f"Starting training for variant: {cfg.variant.name} \
            with resume_enabled={cfg.train.checkpoints.resume_from_checkpoint}")
    ui.info(f"Set MLflow experiment to '{experiment_name}' for training run.")
    
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
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    ui.info("Initializing training components...")

    ui.info(f"Loading datasets from {cfg.data.train_path} and {cfg.data.val_path}")
    train_dataset = GalaxyDataset(labels_df=pd.read_csv(cfg.data.train_path), 
                                  img_dir=cfg.data.img_dir, transform=raw_transforms)
    
    val_dataset = GalaxyDataset(labels_df=pd.read_csv(cfg.data.val_path), 
                                img_dir=cfg.data.img_dir, transform=val_transforms)
    logger.info(
        "Loaded datasets | train_samples={} | val_samples={} | image_dir={}",
        len(train_dataset),
        len(val_dataset),
        cfg.data.img_dir,
    )

    ui.debug(
        f"Loaded datasets | train_samples={len(train_dataset)} \
        | val_samples={len(val_dataset)} | image_dir={cfg.data.img_dir}",
    )

    train_sampler, train_class_weights = train_dataset.make_weighted_sampler()
    logger.info("Weighted sampler prepared | class_weights={}", train_class_weights)
    ui.debug(f"Weighted sampler prepared | class_weights={train_class_weights}")

    train_dataloader = DataLoader(num_workers=cfg.env.num_workers, dataset=train_dataset, 
                                  batch_size=cfg.train.batch_size, shuffle=False, 
                                  sampler=train_sampler,pin_memory=True, 
                                  persistent_workers=True, prefetch_factor=4)
    
    val_dataloader = DataLoader(num_workers=cfg.env.num_workers, dataset=val_dataset, 
                                batch_size=cfg.train.batch_size, 
                                shuffle=False,pin_memory=True,
                                persistent_workers=True, prefetch_factor=4)
    
    #load models and optimizers
    assert cfg.variant.num_classes is not None, \
    f"variant '{cfg.active_variant}' has num_classes=null, cannot use classification model"

    model = GalaxyClassifier_ResNet(num_classes=cfg.variant.num_classes)
    
    if dev != torch.device("cuda"):
        logger.warning("CUDA device not available, training will be performed on CPU. This may be slow.")
        ui.warning("CUDA device not available, training will be performed on CPU. This may be slow.")

    model.to(dev)

    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.learning_rate,
                                  weight_decay=cfg.train.weight_decay) #weight decay

    scheduler = torch.optim.lr_scheduler.\
    CosineAnnealingWarmRestarts(optimizer, 
                                T_0=cfg.train.scheduler.warmup_epochs,
                                eta_min =cfg.train.scheduler.min_lr 
                                ,T_mult=2)
    
    train_criterion = torch.nn.CrossEntropyLoss(label_smoothing=cfg.train.label_smoothing) #weighted for training only
    val_criterion = torch.nn.CrossEntropyLoss() #single weighted for training only
    
    metrics   = make_metrics(cfg.variant.num_classes, dev)
    writer    = SummaryWriter(log_dir=cfg.storage.local.logs.tensorboard_dir / experiment_name)

    ui.debug(f"TensorBoard writer initialized | log_dir={cfg.storage.local.logs.tensorboard_dir / experiment_name}")

    az_checkpoint_dir = cfg.train.checkpoints.checkpoint_restore_path
    chkpt_local_path = Path(cfg.train.checkpoints.local_checkpoint_path) / experiment_name   

    chkpt = CheckpointManager(cfg=cfg,
                                checkpoint_dir_local=chkpt_local_path,
                                checkpoint_dir_azure=az_checkpoint_dir,
                                resume_enabled=resume,
                                resume_run_id = resume_id or cfg.train.checkpoints.resume_run_id)
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
    chkpt_state = chkpt.load_checkpoint(model=model, optimizer=optimizer, 
                                        scheduler=scheduler, device=dev,
                                        blob_file_name= CHKPT_TYPE[chkpt_type], save_filename="best_resume.pt")
    
    best_val_f1_macro = float("-inf")   #-inf, on resume_checkpoint, run fresh so it can beat its own 
    MIN_DELTA = 1e-3    

    cfg_snapshot = {
        "variant":     cfg.active_variant,
        "num_classes": cfg.variant.num_classes,
        "lr":          cfg.train.learning_rate,
        "batch_size":  cfg.train.batch_size,
    }# to log in mlfow

    mlflow.enable_system_metrics_logging()
    ui.trace(f"MLflow system metrics logging enabled | tracking_uri={mlflow.get_tracking_uri()}")
    
    Resume = resume or cfg.train.checkpoints.resume_from_checkpoint

    if Resume:
        experiment_name = f"resume_{experiment_name}"  #resume diff name 
        logger.info(f"Resuming training from checkpoint | experiment_name={experiment_name}")
        
    if chkpt_state.start_epoch >= cfg.train.num_epochs:
        raise ValueError(f"num_epochs={cfg.train.num_epochs} <= start_epoch={chkpt_state.start_epoch}; nothing to train.")
    
    #split run for checkpoints instead of same run in resume  
    
    tags = {"resume": str(bool(Resume)).lower()}          
    if Resume and chkpt_state.resume_run_id:
        tags["parent_run_id"] = str(chkpt_state.resume_run_id)
        
    with mlflow.start_run(run_name=experiment_name, 
                          tags= tags) as run:  #set tag for parents for easy tracking 
        
        run_id = run.info.run_id
        chkpt.run_id = run_id #assign new run_id to checkpoint manager for future saves
        
        logger.info(
            "MLflow run started | run_id={} | resume_run_id={} | start_epoch={} | best_val_loss={}",
            run_id,
            chkpt_state.resume_run_id,
            chkpt_state.start_epoch,
            chkpt_state.best_val_f1_macro,
        )

        ui.info(
            f"MLflow run started | run_id={run_id} | resume_run_id={chkpt_state.resume_run_id} |\
              start_epoch={chkpt_state.start_epoch} | best_val_f1_macro={chkpt_state.best_val_f1_macro}",
        )

        mlflow.log_params({
            "epochs":          cfg.train.num_epochs,
            "batch_size":      cfg.train.batch_size,
            "lr":              cfg.train.learning_rate,
            "weight_decay":    cfg.train.weight_decay,
            "applied_weight_decay": optimizer.param_groups[0]["weight_decay"],
            "grad_clip":       cfg.train.grad_clip_max_norm,
            "label_smoothing": cfg.train.label_smoothing,
            "applied_label_smoothing": train_criterion.label_smoothing,
            "variant":         cfg.active_variant,
            "num_classes":     cfg.variant.num_classes,
            "optimizer":       cfg.train.optimizer,
            "scheduler":       cfg.train.scheduler.type,
            "T_0":             scheduler.T_0,
            "T_mult":          scheduler.T_mult,
            "resumed":           str(bool(Resume)),
            "resume_run_id":     chkpt_state.resume_run_id,
            "start_epoch":         chkpt_state.start_epoch,
            "seed":            cfg.env.seed,
        })
        
        global_step = chkpt_state.start_epoch * len(train_dataloader)
        with ui.train.session(epochs=cfg.train.num_epochs, steps_per_epoch=len(train_dataloader)) as dash: #ui
            
            
            for ep in range(chkpt_state.start_epoch, cfg.train.num_epochs):
                dash.start_epoch(ep)
                # need to use rich for a bar during training
                train_loss, global_step = train_epoch(
                    model, train_dataloader, optimizer, scheduler, train_criterion,
                    writer, ep, cfg.train.num_epochs, global_step,
                    cfg.train.grad_clip_max_norm,ui,dash ,dev,
                )
                
                val_loss, val_metrics = validate_epoch(
                    model, val_dataloader, val_criterion, metrics,
                    ep, cfg.train.num_epochs,ui,dash ,dev,
                    
                )

                val_f1_macro = val_metrics["val_f1_macro"]
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
                    "epoch":            ep + 1,
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



                if val_f1_macro > best_val_f1_macro + MIN_DELTA:  # Check for improvement with a minimum delta
                    best_val_f1_macro = val_f1_macro
                    logger.info("New best validation f1 macro {:.4f} at epoch {}", val_f1_macro, ep + 1)
                    ui.info(f"New best validation f1 macro {val_f1_macro:.4f} at epoch {ep + 1}")

                    chkpt.save(model, optimizer, scheduler, ep, val_loss, val_metrics["val_f1_macro"], 
                            cfg_snapshot, filename="best.pt")
                    try:
                        ckpt_url =chkpt.upload_checkpoint_to_azure(filename="best.pt")#upload immediately after improvement for better restore
                        mlflow.set_tag("checkpoint_url", ckpt_url)
                        mlflow.set_tag("checkpoint_blob_path", f"checkpoints/{run_id}/best.pt")
                        ui.info(f"Best checkpoint synced to Azure | url={ckpt_url}")
                        
                    except Exception as e:
                        logger.error("Azure upload failed: {}", e)
                        ui.warning(f"Azure upload failed (local best.pt saved): {e}")
                
                if cfg.train.checkpoint.save_latest:

                    chkpt.save(model, optimizer, scheduler, ep, val_loss, #save every epoch locally
                        val_metrics["val_f1_macro"], cfg_snapshot, filename="latest.pt")
                
                if (ep + 1) % LATEST_UPLOAD_EPOCHS == 0:
                    chkpt.upload_checkpoint_to_azure(filename="latest.pt")   
                
                if cfg.train.checkpoint.save_cycle:
                    CYCLE_ENDS = cycle_ends(scheduler.T_0, scheduler.T_mult, cfg.train.num_epochs) 
                    # cycle ends for lr have better metrics naturally
                    if (ep + 1) in CYCLE_ENDS:
                        chkpt.save(model, optimizer, scheduler, ep, val_loss,
                                val_metrics["val_f1_macro"], cfg_snapshot, filename=f"cycle_end_ep{ep+1}.pt")
                        chkpt.upload_checkpoint_to_azure(filename=f"cycle_end_ep{ep+1}.pt")
                        ui.info(f"Cycle end checkpoint saved and uploaded: cycle_end_ep{ep+1}.pt")
                        
                dash.end_epoch(val_loss=val_loss, val_acc=val_metrics["val_acc"])
        # after training   
        

        #non verbosed artifact upload for final best checkpoint and report generation
        with ui.status("Finalizing training and uploading artifacts to Azure Blob Storage...", spinner="dots") as status:
            with AzureBlobStorage() as blob:
                # ensure experiment container exists
                logger.info("Ensuring Azure artifact container exists: {}", cfg.storage.cloud.experiments.container)
                ui.debug(f"Ensuring Azure artifact container exists: {cfg.storage.cloud.experiments.container}")
                blob.create_container(name=cfg.storage.cloud.experiments.container)

                # load best checkpoint (support either file path or directory containing best.pt)
                best_candidate = Path(chkpt.checkpoint_dir)
                if best_candidate.is_file():
                    best_ckpt_path = best_candidate
                    ui.trace(f"Best checkpoint is a file: {best_ckpt_path}")
                else:
                    best_ckpt_path = best_candidate / "best.pt"

                best_ckpt = torch.load(best_ckpt_path, map_location=dev)
                model.load_state_dict(best_ckpt["model_state_dict"])
                ui.trace(f"Loaded best weights from: {best_ckpt_path}") 

                logger.info(
                    "Loaded best weights | epoch={} | checkpoint_path={}",
                    best_ckpt["epoch"] + 1,
                    best_ckpt_path,
                )
                ui.info(
                    f"Loaded best weights | epoch={best_ckpt['epoch'] + 1} | checkpoint_path={best_ckpt_path}",
                )

                ui.debug("Registering model to mlflow")
                status.update("Registering model to mlflow")
                register_model(model, val_dataloader)

                status.update("Generating training report")
                ui.debug("Generating training report")
                generate_report(run_id=run_id, output_path=cfg.storage.local.logs.reports_dir / f"{experiment_name}.md")

                status.update("Uploading artifacts to Azure Blob Storage")
                ui.debug("Uploading artifacts to Azure Blob Storage")
                upload_artifact(blob, run_id, experiment_name=experiment_name, cfg=cfg)

        writer.close()
        logger.success("Training complete.")
         
    mlflow.end_run()
    return run_id


# CLI entry
@pipeline.command()
@click.option("--variant","-v", 
              type=click.Choice(["Default","Hubble","Dirlichlet"]), default="Default", 
              help="Training variant to use")
@click.option("--evaluate","-e", 
              is_flag=True, default=False, 
              help="Evaluate the model after training")
@click.option("--resume","-r", 
              is_flag=True, default= None, 
              help="Resume training from the last checkpoint if available")
@click.option("-c","--status","show_config", 
              is_flag=True, default=False, 
              help="Show the configuration before training")
@click.option("-y","--yes", 
              is_flag=True, default=False, 
              help="Automatically confirm all prompts")
@click.option("--resume_run_id","-rid", 
              type=str, default=None, 
              help="Run ID of the checkpoint to resume from")
@click.option("--resume_checkpoint_from",
              type=click.Choice(["Best","Latest"]),default = "Best",
              help="The checkpoint to resume from from best,latest and cycle end checkpoints from blob")

@click.pass_context

def train(ctx, 
          variant:str, 
          evaluate:bool, 
          show_config:bool, 
          resume_run_id:str, 
          resume:Optional[bool], yes:bool,
          resume_checkpoint_from:Literal["Best","Latest"]):

    from src.utils.bootstrap import load_context
    from src.cli.status import ConfigUI
    from rich.prompt import Confirm
    from src.training import validate_setup
                
    from src.utils.logging_config import forward_mlflow_logs_to_rich
    
    import os
    
    ui = ctx.find_object(UI) or UI(verbose=0)  # Retrieve the UI object from the context or create a new one
    
    forward_mlflow_logs_to_rich(console = ui.console)
    
    try:
        
        app_ctx = ctx.obj or load_context(service="ml",std_out=False) #retrive from top level of click cmd in __init__py, fallback to load_context if not present
        cfg= app_ctx.cfg
        
        
        MLFLOW_TRACKING_URI = app_ctx.mlflow_tracking_uri

        if show_config:
            with ui.status("Displaying current configuration...", spinner="dots") as status:
                config_ui = ConfigUI(ui)
                config_ui.show_train_config(cfg, variant, MLFLOW_TRACKING_URI)
                return  # Exit after showing the configuration
            
            
        if yes:
            ui.info("Auto-confirm enabled. Proceeding with training without user confirmation.")
        
        else:
            if Confirm.ask("Are you sure you want to start training? \
                        This may take a long time and consume significant resources.", default=False):
                pass
            
            else:
                ui.info("Training aborted by user.")
                return

        #slower importing time so modules were put below    
        with ui.status("Starting training process...", spinner="dots") as status:

            status.update("Importing required modules...")

            import mlflow

            from ml.pipeline.eval import evaluate as evaluate_command #for click ctx


            status.update("Validating setup...")
            if not validate_setup(required_paths=[cfg.data.img_dir,cfg.data.train_path,cfg.data.val_path], 
                        mlflow_tracking_uri=MLFLOW_TRACKING_URI,
                        
                    create_dirs =   [cfg.storage.local.checkpoints.dir,
                    cfg.storage.local.logs.reports_dir,
                    cfg.storage.local.logs.tensorboard_dir]):
                
                
            
                ui.error("setup validation failed. Please check the logs for details at:")
                if app_ctx.log_path is not None:
                    ui.print(ui.link_path(app_ctx.log_path))
                return

            status.update("starting training for variant: {}".format(variant))
    
                
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        if variant == "Default":
            mlflow.set_experiment(cfg.variant.experiment.root)
            run_id = train_default(cfg,ui,resume=resume,resume_id=resume_run_id,chkpt_type=resume_checkpoint_from)
            ui.success("Training complete." ,style = "bold spring_green2")
            

            if evaluate:
                ui.rule("Evaluating the trained model...", style="bold spring_green2")
                ui.info("Starting evaluation of the trained model...")
                
                evaluate_command.callback(run_id=run_id, variant=variant, model_version=None,
                    min_improvement=0.005, metric_name="f1_macro", alias="Champion", 
                    model_source="mlflow")
    
                
    except Exception as e:
        ui.error(f"An error occurred during training: {e}",show_execption=True)
        raise  



def train_hubble():
    pass

def train_dirilchled():
    pass


if __name__ == "__main__":
    train()
