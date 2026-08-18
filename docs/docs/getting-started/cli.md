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

## Basic Commands

running a training script. 
```sh

galax pipeline train --variant="default"

#with verbose

galax -vv pipeline train --variant="default" # level 2
```

```sh
$ galax -vv cache import --checkpoint_path="models\cached\9e2fb45714864e35bf92fc2223d7e7fe.pt" --variant="Default" --num_classes=5 --model_name="default" --local_cache_dir="tests" --metadata_save_file_path="models.json" --metric_name="test" --metric=0.1
```