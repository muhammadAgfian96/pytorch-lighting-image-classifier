# FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
# FROM pytorch/pytorch:latest
RUN apt update && apt install --no-install-recommends -y zip htop screen libgl1-mesa-glx
RUN apt update && apt install -y libsm6 libxext6
RUN apt-get install -y libxrender-dev
# RUN pip install --no-cache-dir torch==2.0.1 torchmetrics==1.0.1 torchvision==0.15.2 pytorch-cuda=11.7 -c pytorch -c nvidia
COPY requirements.txt /depedencies/requirements.txt
RUN python -m pip install --upgrade pip
RUN pip install --no-cache -Ur /depedencies/requirements.txt 
ENV PYTHONPATH=/workspace
WORKDIR /workspace