FROM python:3.9.16 as python
FROM nvidia/cuda:12.0.1-cudnn8-runtime-ubuntu20.04
COPY --from=python /usr/local/ /usr/local/

RUN apt-get update \
    && apt-get install -y \
    wget \
    git \
    sudo \
    && rm -rf /var/lib/apt/lists/*

RUN pip install pip --upgrade
COPY pytorch_gpu_project/.devcontainer/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# install python tools
COPY common/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt