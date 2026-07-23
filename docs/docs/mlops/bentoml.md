# Setup

## Build & Containerize
Firstly build the bentoml service using using the following command:

``` bash
bentoml build --version 0.6.0 --containerize

```
or 
``` bash
source(./setup.sh && bentoml_build)
```
## Response

``` json


```

## Run the service
Once the service is built, you can run it using the following commannd in the docker compose: 

``` bash
docker compose up bentoml
```

!!! warning "Make sure to run the bentoml service first"
    The bentoml service should be running before the api service is started. This is because the api service depends on the bentoml service to load the model and run inference. If the bentoml service is not running, the api service will fail to start.





# Structure
I used service and runner structure that seperates the model loading and inference from the main service. This is done to make the code more modular and easier to maintain. The service is responsible for handling the requests and responses, while the runner is responsible for loading the model and running the inference. This structure also allows for easy scaling of the service by adding more runners as needed.

![Bentoml](../assets/bentoml.png)

## Services




---
## Improvements


!!! info "Future Improvements"
    - Adding backup to cache to pull model from azure blob if mlflow is down. This will make the service more robust and reliable.
    - Using a more robust model loading mechanism that can handle different model formats and versions. This will make the service more flexible and easier to maintain.
