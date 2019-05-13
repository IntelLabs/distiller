FROM pytorch/pytorch:0.4_cuda9_cudnn7

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

ADD . /workspace/pytorch
RUN pip install -r /workspace/pytorch/requirements.txt

WORKDIR /workspace/pytorch
