# Training and evaluation

This page gives the shortest repeatable workflow for the ML pipeline. Run
commands from the repository root with the project environment active.

## 1. Check configuration

```bash
galax status
galax pipeline train --status
```

The status command verifies the resolved paths and service configuration before
the training command starts.

## 2. Prepare data

Run preprocessing and splitting when the configured raw label files need to be
rebuilt:

```bash
galax pipeline preprocess \
  --run_preprocessing \
  --preprocess_type classic \
  --run_split \
  --variant Default
```

The output split files are written to the configured split directory.

## 3. Train

```bash
galax -vv pipeline train --variant Default --yes
```

Use `--resume` and optionally `--resume_run_id` to continue a previous run:

```bash
galax pipeline train \
  --variant Default \
  --resume \
  --resume_run_id RUN_ID \
  --resume_checkpoint_from Best \
  --yes
```

Add `--evaluate` to evaluate after training.

## 4. Evaluate

Evaluate a model recorded in MLflow:

```bash
galax pipeline evaluate \
  --run_id RUN_ID \
  --variant Default \
  --model_source mlflow
```

Evaluate a local model instead:

```bash
galax pipeline evaluate \
  --run_id local-check \
  --model_source local \
  -m models/checkpoint.pt
```

The local source requires `-m`; the MLflow source must not be
combined with it.

## 5. Export or inspect metrics

```bash
galax pipeline load-metrics \
  --run_id RUN_ID \
  --metric_names f1_macro \
  --metric_names accuracy \
  --save_path reports/metrics.csv
```

For all options, see the [CLI command reference](index.md).
