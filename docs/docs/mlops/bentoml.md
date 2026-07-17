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

## Run the service
Once the service is built, you can run it using the following commannd in the docker compose: 

``` bash
docker compose up bentoml
```

!!note: 




d
# Structure
I used service and runner structure that seperates the model loading and inference from the main service. This is done to make the code more modular and easier to maintain. The service is responsible for handling the requests and responses, while the runner is responsible for loading the model and running the inference. This structure also allows for easy scaling of the service by adding more runners as needed.

![Bentoml](/docs/assets/bentoml.png)

##





# Improvements

