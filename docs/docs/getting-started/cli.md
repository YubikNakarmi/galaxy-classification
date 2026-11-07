## Intro

!!! Warning
    To be able to use CLI you need to install the editable pacakge of use uv. Also ensure that you have virtual environment activated as it might not work if you have multiple python versions installed.

```bash
uv sync --frozen #auto installs

#or
pip install -e . #old format
```

The main comamnd for this project is

```sh 
galax --help 
```
You should see something like this:

![Root of CLI](../assets/cli/main.png){ width="400" height="400" align="center" }

> More detailed architecute of cli on [CLI](/docs/docs/development/cli.md#overview) documentation.

For the complete command reference, including pipeline, cache, cloud, setup,
and BentoML commands, see the [CLI documentation](../cli/index.md).

## Basic commands
### Status
```sh
galax status
```

![Status](../assets/cli/status.png)

### Importing a model
```sh
$ galax -vv cache import \
--checkpoint_path="models\cache\9e2fb45714864e35bf92fc2223d7e7fe.pt" \
--variant="Default" --num_classes=5 --model_name="galaxy-classification-default" \
--local_cache_dir="models/cache" --metadata_save_file_path="models/cache/model.json" \
--metric_name="test" --metric=0.1
```

![Import](../assets/cli/save_model.png)

### running a training script. 

Verifying the configs
```sh
galax pipeline train -c
```
![Status](../assets/cli/training_status.png)


Running Training

```sh
galax pipeline train --variant "Default"
#with verbose
galax -vv pipeline train --variant "Default"
# level 2
```

![Training](../assets/cli/training.png)

### View logs

```sh
galax log show -n 50 -s "ml"
```

![result](../assets/cli/logs.png)
