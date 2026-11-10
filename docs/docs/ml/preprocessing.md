## Preprocessing Script

The exection of script can be perfomed with:

```sh
python ml/preprocess.py --run_preprocessing=True --run_split=True

```
The `--run_preprocessing` flag controls whether the preprocessing step is executed, while the `--run_split` flag controls whether the data splitting step is executed. This allows for flexibility in the workflow, enabling you to run only the preprocessing or splitting steps as needed.

## Run Preprocessing
This flag is used to control whether the preprocessing step is executed. When set to `True`, the script will perform the necessary preprocessing steps on the dataset, such as cleaning, transforming, and encoding the data. This may include tasks like handling missing values, normalizing features, and encoding categorical variables.

It requires raw files `labels.csv` and `mapping.csv`.

## Run Split
used to run the data splitting step, which divides the preprocessed dataset into training, validation, and test sets based on the specified split ratios. This requires

`labeled_data_1.csv` file which is the output of preprocessing step. The split ratios can be configured in the `ml.yaml` config file under the `preprocessing` section as shown below:

``` yaml title="Preprocessing split configuration in ml.yaml hl_lines="7"
## Running

the preprocessing script is located in `ml/preprocess.py` and is responsible for loading the dataset, applying necessary transformations, and splitting the data into training and validation sets. The script uses the `GalaxyDataset` class to handle the dataset and the `Label_Processing_Classic` class for label preprocessing. The processed data is then saved for use in training the model.

This script can be run though dvc pipline using the following command:

```sh

dvc repro preprocess

```
This run can be configured in `dvc.yaml` file, where you can specify the dependencies and outputs of the preprocessing stage.


note that the `--run_preprocessing` and `--run_split` flags can be used to control whether the preprocessing and splitting steps are executed, allowing for flexibility in the workflow.
 
The default commands in yaml file are: 

```yaml title="Preprocessing stage in dvc.yaml"
preprocess:
    cmd: python ml/preprocess.py --run_preprocessing=true --run_split=true
    deps:
      - ml/preprocess.py
      - ml/variants.yaml
      - data/raw/mapping.csv
      - data/raw/labels.csv
    outs:
      - data/preprocessed/
```
---

## Preprocessing Configurations
The preprocessing configurations ar defined in `configs/ml.yaml`.

### Preprocessing type
There are three types of preprocessing that can be performed on labels: `Classic`, `Hubble`, and `Dirichlet`. As defined in module `src.dataset.label_preprocessing`, each type of preprocessing applies different transformations to the labels. The type of preprocessing to be applied can be specified in the `ml.yaml` config file under the `preprocessing` section as shown below:


```yaml title="Preprocessing type configuration in ml.yaml hl_lines="1 4" 
active_variant: Default
Variants:
    Default:
        preprocessing_type: Default
```

the ```active_variant``` key is used to specify which variant of preprocessing to be applied. The ```Variants``` type itself can be refernced in the preprocessing script using the active varaint type for the model. 


Each variant has ```split``` level which can be used to configure the split ratios for training. 

``` yaml title="Preprocessing split configuration in ml.yaml hl_lines="7"
variants:
  Default:
    preprocessing: Default
    num_classes: 5
    label_file: data/preprocessed/labels/labeled_data_1.csv

    data:
      split:
        train_size: 0.70
        val_size: 0.15
        test_size: 0.15
      stratify: true
```

---

### Directory configuration
Ultimately the dir in which file is saved is configured in ```data``` along with the raws data paths

```yaml title ="Preprocessing output directory configuration in ml.yaml hl_lines="2 6"
data:
  raw_paths: 
    - labels: data/raw/labels.csv
    - mappings: data/raw/mapping.csv
    
  preprocessed_dir: data/preprocessed/
  split_dir: data/preprocessed/split/
```