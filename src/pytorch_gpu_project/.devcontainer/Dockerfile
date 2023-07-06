FROM python:3.9.16 as python
FROM nvidia/cuda:12.0.1-cudnn8-runtime-ubuntu20.04
COPY --from=python /usr/local/ /usr/local/

RUN apt-get update \
    && apt-get install -y \
    wget \
    git \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# TODO: create non-root user and set the default user but make sure user can save files on host, for example make the user the same as user from host but not root?
#ARG USERNAME=devuser
#ARG USER_UID=1000
#ARG USER_GID=$USER_UID
#RUN groupadd --gid $USER_GID $USERNAME \
#    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
#    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
#    && chmod 0440 /etc/sudoers.d/$USERNAME 
#USER $USERNAME

RUN pip install pip --upgrade
COPY pytorch_gpu_project/.devcontainer/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# install python tools
COPY common/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt