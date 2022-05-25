# sudo docker build -t watermark-detector:v0 -f Dockerfile .
FROM nvidia/cuda:10.1-cudnn7-devel

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y \
	build-essential \
	libopencv-dev \
	python3-pip \
	git \
    && apt-get clean && rm -rf /tmp/* /var/tmp/*

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r requirements.txt
RUN python3 -m pip install torch==1.8 torchvision==0.9 -f https://download.pytorch.org/whl/cu101/torch_stable.html
RUN python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git'