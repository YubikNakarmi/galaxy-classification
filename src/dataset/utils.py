
import matplotlib.pyplot as plt
import mlflow
from pathlib import Path






def generate_report(run_id: str, output_path: Path):


    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    params = run.data.params
    metrics = run.data.metrics

    # pull per-epoch history for curves
    loss_history = client.get_metric_history(run_id, "val_loss")
    f1_history = client.get_metric_history(run_id, "val_f1_macro")

    # plot curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot([m.step for m in loss_history], [m.value for m in loss_history])
    axes[0].set_title("Val Loss")
    axes[1].plot([m.step for m in f1_history], [m.value for m in f1_history])
    axes[1].set_title("Val F1 Macro")
    curves_path = output_path.parent / "curves.png"
    fig.savefig(curves_path)

    md = f"""# Experiment: {params.get('variant')} — {run.info.run_name}

## Parameters
| Key | Value |
|-----|-------|
{"".join(f"| {k} | {v} |\n" for k, v in params.items())}

## Metrics
| Metric | Value |
|--------|-------|
| val_loss | {metrics.get('val_loss', 'N/A'):.4f} |
| val_acc | {metrics.get('val_acc', 'N/A'):.4f} |
| val_f1_macro | {metrics.get('val_f1_macro', 'N/A'):.4f} |
| val_roc_auc | {metrics.get('val_roc_auc', 'N/A'):.4f} |

## Training Curves
![curves](curves.png)
"""
    output_path.write_text(md)