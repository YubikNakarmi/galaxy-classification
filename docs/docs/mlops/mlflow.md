## Configuration

MLflow is configured through `ml.yaml` in `/configs` directory.

```yaml title="/configs/ml.yaml" hl_lines="4"
variants:
    Default:
        experiment:
            root: galaxy-classification-default-experiments # (1)

```

1. This configures naming for training/eval experiments names

## Experiments
The experiment names are configured as 

## Registry