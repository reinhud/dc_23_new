# Dockerized TIMM Computer Vision Backend & FastAPI 

This is a project set up for the course "Data Challenges SoSe23" at Goethe Uni.
It features a backend for a computer vision leveraging the possability of
easily extending the 'timm' to easily spin up some SOTA deep learning models.
In this project we use these models for transfer learning to classify ancient coins. 
The application is dockerized to make development easy and the service is exposed via FastAPI 
endpoints allowing the models to be interacted with and train and infer with http requests.

## Contents
- 

## Introduction and Overview


#### Features


## Getting Started

#### Dependencies
- [VsCode](https://code.visualstudio.com/)
    - [VsCode Remote - Containers Extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
- [Docker Desktop](https://code.visualstudio.com/docs/devcontainers/containers#_installation)
  

#### How to set up
- Start only Pytorch Project:
  - From command pallet(strg+P)``` Dev Containers: Open Folder in Container``` and select ```pytorch_service``` folder
- With API:
  - From command pallet: ```Dev Containers: Open Folder in Containe```r and select ```src``` folder
    - from terminal in the container run ```uvicorn api.src.fastapi_server:app --reload --host 0.0.0.0 --port 80```
      to start the FastAPI server
    - in browser enter http://localhost:8000/docs for Swagger endpoint documentation
  
## Directory Structure


## CI Pipeline


## Licence


## Acknowledgments

Inspiration, usefull repos, code snippets, etc.
