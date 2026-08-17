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