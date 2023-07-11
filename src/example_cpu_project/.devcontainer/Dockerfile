FROM python:3.9.16

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