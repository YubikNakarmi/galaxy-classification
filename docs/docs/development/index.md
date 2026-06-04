# Development

## Configs
All the system configs are located in `configs/` directory. There are `backend.yaml`, `core.yaml` and `ml.yaml` configs files. 


### Core Configs
The `core.yaml` file contains all the core configurations for the system. It is used to store the parameters for the core functionalities of the system such as data processing, model training and evaluation. It also contains the paths to the datasets, models and other resources used in the system. For example loggin can be configured in `core.yaml` as shown below:

``` yaml title="Logging snippet of core.yaml"
.......
logging:

  app:
    level: INFO
    file: logs/system/app.log
    rotation: "10 MB"
    retention: "7 days"
    compression: zip
  
  ml:
    level: INFO
    file: logs/system/ml.log
    rotation: "10 MB"
    retention: "7 days"
    compression: zip
.......
```


### ML Configs
The `ml.yaml` file contains all the machine learning related configurations. It is used to store the parameters for training, evaluation and inference of the machine learning models. It also contains the paths to the datasets, models and other resources used in the machine learning pipeline. 



