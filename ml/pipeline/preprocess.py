
from __future__ import annotations

from pathlib import Path

import click
from . import pipeline


#setup logging and config load
root = Path(__file__).resolve().parent.parent




#splitting data 

#no typing to lower the latnency of click cli command and avoid loading config multiple times in each command
def split(df,cfg):
    import pandas as pd
    from sklearn.model_selection import train_test_split

    seed = cfg.env.seed
    active_variant = cfg.active_variant
    test_size = cfg.variants[active_variant].data.split.test_size

    train_size = cfg.variants[active_variant].data.split.train_size
    val_size = cfg.variants[active_variant].data.split.val_size


    ids = df.drop_duplicates("asset_id")[["asset_id", "label"]]
    
    train_val_ids, test_ids = train_test_split(
    ids,
    test_size=test_size,
    stratify=ids["label"],
    random_state=seed)


    train_ids, val_ids = train_test_split(
    train_val_ids,
    test_size=val_size/(train_size + val_size),  # Adjust val_size to be a proportion of the remaining data
    stratify=train_val_ids["label"],
    random_state=seed)

  
    
    return train_ids, val_ids, test_ids



@pipeline.command()
@click.option('--run_preprocessing',default= False, help='Whether to run preprocessing or not')
@click.option('--preprocess_type', default='classic', help='Type of preprocessing to run (e.g., classic, hubble,dirilichet)')
@click.option('--run_split', default=False, help='Whether to run data splitting or not split defined in ml.yaml')
@click.option('--variant', default="Default", help='Variant of the experiment to preprocess.',type=click.Choice(["Default", "Hubble", "Dirichlet"]))
@click.pass_context #ctx from top level
def preprocess(ctx, run_preprocessing,run_split,preprocess_type):
    import pandas as pd
    from loguru import logger

    from src.dataset.label_preprocessing import Label_Processing_Classic
    from src.cli.cli_bootstrap import load_context
     #custom contexxt to avaoid multi load
    app_ctx = ctx.obj or load_context(service="ml")#retrive from top level of click cmd in __init__py

    cfg = app_ctx.cfg

  
    #core params
    active_variant = cfg.active_variant    
    # dirs and paths for preprocessing and splitting
    split_dir = cfg.data.split_dir
    split_dir.mkdir(parents=True, exist_ok=True)

    label_file = cfg.variants[active_variant].label_file

    preprocessed_dir = str(cfg.data.preprocessed_dir)

    mapping = pd.read_csv(str(cfg.data.raw_paths.mappings))

    labels = pd.read_csv(str(cfg.data.raw_paths.labels))

    if run_preprocessing:

        if mapping is None or labels is None:
            logger.exception("Mapping and labels file paths must be provided in the ml.yaml configuration.")
            return
        
        if preprocess_type == 'Default':
            logger.info("Running classic label preprocessing...")

            merge = pd.merge(mapping, labels, left_on='objid', right_on='dr7objid')

            label_processor = Label_Processing_Classic(df=merge, save_path=preprocessed_dir)
            deb = label_processor.extract_debiased_df()
            
            label_processor.find_voting_columns()
            processed_labels = label_processor.assign_label(deb,series=False)

            processed_labels.to_csv(Path(preprocessed_dir) / 'processed_labels.csv', index=False)
            logger.info(f"Processed labels saved to {Path(preprocessed_dir) / 'processed_labels.csv'}")
        else:
            logger.warning(f"Preprocessing type '{preprocess_type}' is not recognized. Skipping preprocessing.")
        #other preprocessing is progresss

    
    logger.info(f"Loading label file from {label_file}")

    if run_split:
        if not Path(label_file).exists():
            logger.exception(f"Label file '{label_file}' does not exist. Please check the path and try again.")
            return
        
        labels_df = pd.read_csv(label_file)
        splits = split(labels_df, cfg)

        train_ids, val_ids, test_ids = splits
        for split_name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
            split_df = labels_df[labels_df["asset_id"].isin(ids["asset_id"])]

            split_path = Path(split_dir) / f"{split_name}_split.csv"
            split_df.to_csv(split_path, index=False)
            logger.info(f"{split_name.capitalize()} split saved to {split_path}")
    
# pythron preprocess.py --run_preprocessing=true --preprocess_type="classic" --run_split=true
if __name__ == "__main__":
    preprocess()