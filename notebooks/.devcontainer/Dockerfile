FROM python:3.9.16

# make all python tools installed by pip accesible
ENV PATH=$PATH:/home/$USERNAME/.local/bin
RUN pip install pip --upgrade
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt