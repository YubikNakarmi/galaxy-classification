# Galaxy Classification System

## Movitation
The reason for me working of project is due to multitude of factors. Firstly, I was always interested in astronomy and Neural networks and I wanted to buiild something regarding Astronomy. Then there also the motivation to build something better than my lat project [AQI_predictor](https://github.com/YubikNakarmi/Aqi_predictor), which is pretty messy. I wanted to contiune to learn to build better systems wich more cleaner and maintainable code with also better organizations and potability. With as miminal use of AI as possible, except is somethign like comments, docstrings or documentation .Lastly, I also wanted to learn new tools for DevOPS, Mlops, and ML in general I leared tools like:

- Pytorch
- FastAPI
- BentoML
- Promethus
- Grafana
   and more


## Introduction
This project was focused on building a custom ResNet model for classifying galaxy images using Galaxy Zoo 2 dataset[@Willett_2013]. This projct not only focus on buidling and training the model, but also on the entire pipeline, including data preprocessing, model architecture design, training strategies, and evaluation metrics. Furthermore those processes are done with MLOps practices in mind, ensuring reproducibility and scalability. 

Lastly it is served using an API built with FastAPI, allowing for easy integration and deployment. This projects expands on [@Zhu_2019] general architecture of ResNet and implements a whole end to end pipeline for galaxy classification. The project is structured as a microservices architecture, with each service responsible for a specific aspect of the project. Each service is deployed through **containers** which can be scaled independently using tools like Docker Swarm or Kubernetes.

## Objectives
The main objective of this project was to have a full indepth learning experice of a machine/deep learning architecture, from data preprocessing to model deployment. The project aimed to provide a comprehensive understanding of the entire machine learning pipeline, including data preprocessing, model architecture design, training strategies, and evaluation metrics. Additionally, the project aimed to implement MLOps practices to ensure reproducibility and scalability of the model.

## Microservices Architecture
The project is structured as a microservices architecture, with each service responsible for a specific aspect of the project. Each service is deployed through **containers** which can be scaled independently using tools like Docker Swarm or Kubernetes. The services are as follows:

- `api` — FastAPI service that serves the model and provides an endpoint for predictions.
- `bentoml` — BentoML service that handles model serving and deployment.
- `mlflow` — MLflow service that tracks experiments, logs metrics, and manages model versions.

## Model and metrics
Model repository can be found on [Hugging Face Repo](https://huggingface.co/YubikNakarmi/GalaxyZooCNN)
   