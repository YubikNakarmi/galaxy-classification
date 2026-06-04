# Logging

Logging is done through loguru library. The logging configuration is defined in `core.yaml` file under the `logging` section. There are two loggers defined, one for the application logs and another for the machine learning logs. The loggers can be configured with different levels, file paths, rotation, retention and compression settings.

## Configs

``` yaml title="Logging snippet of core.yaml"

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
```
The loggin can be configured in `core.yaml` as shown above. The `app` logger is used for logging the application logs and the `ml` logger is used for logging the machine learning logs. The loggers can be configured with different levels, file paths, rotation, retention and compression settings. The log files will be created in the `logs/system/` directory. The log files will be rotated when they reach 10 MB in size, retained for 7 days and compressed using zip format (default config). 

## File Stream

`logs/system/app.log` and `logs/system/ml.log` are the file streams for the application and machine learning logs respectively. The logs will be written to these files based on the configuration defined in `core.yaml`. The log files will be created if they do not exist and will be rotated, retained and compressed based on the settings defined in the configuration. 

`src/utils/logging_config.py` parses the yaml and setups local file stream in `logs/system`. To utilize logging: 

``` py hl_lines="6 7"
from loguru import logger
from src.utils.yaml_utils import load_yaml
from src.utils.logging_config import setup_app_logging


log_config = config = load_yaml(root / 'configs' / 'core.yaml') # loads config yaml file 
setup_app_logging(log_config) #pass yaml file to parser and setup app mode logging

logger.info("Hello :)") # example log message
```

Result in `logs/system/app.log`:

``` txt title="Example log message in app.log"
2026-06-04 20:42:36.542 | INFO     | __main__:<module>:122 - Hello :)

```