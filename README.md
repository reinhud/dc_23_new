# Dockerized TIMM Computer Vision Backend & FastAPI 
Experimenting with setting up a backend computer vision classification service with huggingfaces pytorch-image-models (TIMM).

## Contents
- [1. Introduction and Overview](#1-introduction-and-overview)
- [2. Getting Started](#2-getting-started)
- [3. Directory Structure](#3-directory-structure)
- [4. CI Pipeline](#4-ci-pipeline)
- [5. Licence](#5-licence)
- [6. Acknowledgments](#6-acknowledgments)



## 1. Introduction and Overview
This is a project set up for the course "Data Challenges SoSe23" at Goethe Uni.
It features a backend for a computer vision leveraging the possability of
easily extending the 'timm' to easily spin up some SOTA deep learning models.
In this project we use these models for transfer learning to classify ancient coins. 
The application is dockerized to make development easy and the service is exposed via FastAPI 
endpoints allowing the models to be interacted with and train and infer with http requests.

#### Features
- backend for flexible transfer learning and inference using timm models
- FastAPI app to expose backend


## 2. Getting Started

#### Dependencies
- [VsCode](https://code.visualstudio.com/)
    - [VsCode Remote - Containers Extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
- [Docker Desktop](https://code.visualstudio.com/docs/devcontainers/containers#_installation)
  

#### How to set up dev environment
- Install and Setup dependancies as described in their documentations
   
**Start only Pytorch Project:** 
- Open VsCode command palette(strg+p)
- run ``` Dev Containers: Open Folder in Container```   
- select ```pytorch_service``` folder       

**With API:**   
- Open VsCode command palette(strg+p)
- run ```Dev Containers: Open Folder in Container```   
- select ```src``` folder  
- from terminal in the container run: ```uvicorn api.src.fastapi_server:app --reload --host 0.0.0.0 --port 80``` to start server    
- In browser enter http://localhost:8000/docs for Swagger endpoint documentation
  
## 3. Directory Structure


## 4. CI Pipeline


## 5. Licence
This project is licensed under the MIT licence - see LICENSE file for details.

## 6. Acknowledgments

Inspiration, usefull repos, code snippets, etc.
- Transfer learning with timm models and pytorch: https://www.kaggle.com/code/hinepo/transfer-learning-with-timm-models-and-pytorch#Dataset-class
- TIMM: https://github.com/huggingface/pytorch-image-models/tree/main
- Pytorch Accelerated: https://github.com/Chris-hughes10/pytorch-accelerated/tree/main/pytorch_accelerated
- dstoolkit-devcontainers: https://github.com/microsoft/dstoolkit-devcontainers
- timmdocs: https://timm.fast.ai/
- Getting Started with PyTorch Image Models (timm): A Practitioner’s Guide: https://towardsdatascience.com/getting-started-with-pytorch-image-models-timm-a-practitioners-guide-4e77b4bf9055
- FastAPI: https://fastapi.tiangolo.com/
