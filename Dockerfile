# FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime
FROM python:3.10-slim
# FROM pytorch/pytorch:latest
RUN apt update && apt install --no-install-recommends -y zip htop screen libgl1-mesa-glx
RUN apt update && apt install -y libsm6 libxext6
RUN apt-get install -y libxrender-dev

COPY requirements.txt /depedencies/requirements.txt
RUN python -m pip install --upgrade pip
RUN pip install --no-cache -Ur /depedencies/requirements.txt 
ENV PYTHONPATH=/workspace
WORKDIR /workspace