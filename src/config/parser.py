from __future__ import annotations
from .schema import *

def _parse_env(raw: dict) -> EnvConfig:
    e = raw["env"]
    return EnvConfig(
        name=e["name"],
        python_version=str(e["python_version"]),
        seed=int(e["seed"]),
        device=e["device"],
        num_workers=int(e["num_workers"]),
    )

def _parse_variant(name: str, raw: dict, root: Path) -> VariantConfig: #pass variant
    data_raw = raw.get("data", {})
    split_raw = data_raw.get("split", {})
    exp_raw = raw.get("experiment", {})
    mlflow_raw = raw.get("mlflow", {})
    model_raw = raw.get("model")
 
    model = None
    if model_raw:
        model = ModelConfig(
            architecture=model_raw["architecture"],
            pretrained=bool(model_raw.get("pretrained", False)),
            K=int(model_raw.get("K", 2)),
            N=int(model_raw.get("N", 2)),
            conv1_stride=int(model_raw.get("conv1_stride", 2)),
            input_size=int(model_raw.get("input_size", 128)),
        )

    return VariantConfig(

    name=name,
    preprocessing=raw["preprocessing"],
    num_classes=raw.get("num_classes"),
    label_file=root / raw["label_file"],
    dirichlet_branches=raw.get("dirichlet_branches"),

    data=VariantDataConfig(
        split=SplitConfig(
            train_size=float(split_raw.get("train_size", 0.7)),#default vals
            val_size=float(split_raw.get("val_size", 0.15)),
            test_size=float(split_raw.get("test_size", 0.15)),
        ),
        stratify=bool(data_raw.get("stratify", True)),
    ),

    experiment=ExperimentConfig(
        root=exp_raw.get("root", ""),
        auto_timestamp=bool(exp_raw.get("auto_timestamp", True)),
        use_variant_name=bool(exp_raw.get("use_variant_name", True)),
    ),

    mlflow=VariantMLflowConfig(
        model_name=mlflow_raw.get("model_name", ""),
    ),

    model=model,
)

def _parse_data(raw: dict, root: Path) -> DataConfig:

    d = raw["data"]
    norm = d.get("normalize", {})
    return DataConfig(
        img_dir=root / d["img_dir"],
        preprocessed_dir=root / d["preprocessed_dir"],
        split_dir=root / d["split_dir"],
        train_path=root / d["train_path"],
        val_path=root / d["val_path"],
        eval_path=root / d["eval_path"],
        input_size=int(d.get("input_size", 128)),

        normalize=NormalizeConfig(
            mean=list(norm.get("mean", [0.1376, 0.1276, 0.1187])),
            std=list(norm.get("std", [0.1728, 0.1586, 0.1488])),
        ),
        raw_paths=RawDataConfig(
            labels=root / d["raw_paths"][0]["labels"],
            mappings=root / d["raw_paths"][1]["mappings"],
        ),
    )
 
 
def _parse_storage(raw: dict, root: Path) -> StorageConfig:
    s = raw["storage"]
    loc = s["local"]
    cld = s["cloud"]
    logs = loc["logs"]
    return StorageConfig(
        local=LocalStorageConfig(
            logs=LocalLogsConfig(
                dir=root / logs["dir"],
                tensorboard_dir=root / logs["tensorboard_dir"],
                system_dir=root / logs["system_dir"],
                reports_dir=root / logs["reports_dir"],
            ),
            checkpoints=LocalCheckpointsConfig(
                dir=root / loc["checkpoints"]["dir"],
                model_cache_dir=root / loc["checkpoints"]["model_cache_dir"]
            ),
            predictions=root / loc["predictions"]["dir"],
        ),
        cloud=CloudStorageConfig(
            dvc=CloudContainerConfig(container=cld["dvc"]["container"]),
            experiments=CloudContainerConfig(container=cld["experiments"]["container"]),
            logs=CloudLogsConfig(
                container=cld["logs"]["container"],
                dir=cld["logs"]["dir"],
            ),
            tensorboard=CloudTensorboardConfig(
                dir=cld["tensorboard"]["dir"],
            ),
            checkpoints=CloudCheckpointsConfig(
                container=cld["checkpoints"]["container"],
                dir=cld["checkpoints"]["dir"],
            ),
            predictions=CloudPredictionsConfig(
                container=cld["predictions"]["container"],
                dir=cld["predictions"]["dir"],
            ),
        ),
    )
 
 
def _parse_train(raw: dict, root: Path) -> TrainConfig:
    t = raw["train"]
    ckpt = t["checkpoints"]
    sched = t.get("scheduler", {})
    return TrainConfig(
        checkpoints=CheckpointTrainConfig(
            resume_from_checkpoint=bool(ckpt["resume_from_checkpoint"]),
            local_checkpoint_path=(root / ckpt["local_checkpoint_path"]) if ckpt.get("local_checkpoint_path") else None,
            checkpoint_restore_path=ckpt["checkpoint_restore_path"] if ckpt.get("checkpoint_restore_path") else None,
            checkpoint_resume_hard=bool(ckpt["checkpoint_resume_hard"]),
            run_id=ckpt.get("run_id"),
        ),
        
        batch_size=int(t["batch_size"]),
        num_epochs=int(t["num_epochs"]),
        learning_rate=float(t["learning_rate"]),
        weight_decay=float(t.get("weight_decay", 1e-4)),
        optimizer=t.get("optimizer", "AdamW"),
        grad_clip_max_norm=float(t.get("grad_clip_max_norm", 1.0)),
        label_smoothing=float(t.get("label_smoothing", 0.1)),
        scheduler=SchedulerConfig(
            type=sched.get("type", "cosine_warmup_restarts"),
            warmup_epochs=int(sched.get("warmup_epochs", 10)),
            min_lr=float(sched.get("min_lr", 1e-6)),
            T_0=int(sched["T_0"]) if sched.get("T_0") is not None else None,
            T_mult=int(sched["T_mult"]) if sched.get("T_mult") is not None else None,
        ),
    )
 
 
def _parse_mlflow(raw: dict) -> MLflowConfig:
    m = raw.get("mlflow", {})
    return MLflowConfig(
        backend=m.get("backend", "sqlite"),
        artifact_store=m.get("artifact_store", "azure"),
        experiment_name=m.get("experiment_name", "galaxy-classification"),
    )
 
 
def parse_config(raw: dict, root: Path) -> RootConfig:
    """
    Convert a validated raw config dict into a RootConfig object.
 
    Args:
        raw: validated dict from loader + validator
        root: project root Path (for resolving relative paths)
    """
    payload = dict(raw)
    payload["variants"] = {
        name: {**variant, "name": name}
        for name, variant in raw.get("variants", {}).items()
    }

    raw_paths = raw["data"]["raw_paths"]
    payload["data"] = {
        **raw["data"],
        "raw_paths": {
            "labels": raw_paths[0]["labels"],
            "mappings": raw_paths[1]["mappings"],
        },
    }
    
    try:
        cfg = RootConfig.model_validate(payload).resolve_paths(root)
        return cfg
    except Exception as e:
        print(f"Error parsing config: {e}")
        raise