FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime
RUN apt update && apt install --no-install-recommends -y zip htop screen libgl1-mesa-glx
RUN apt update && apt install -y libsm6 libxext6
RUN apt-get install -y libxrender-dev

COPY requirements.txt /depedencies/requirements.txt
RUN python -m pip install --upgrade pip
RUN pip install --no-cache -r /depedencies/requirements.txt 
WORKDIR /workspace