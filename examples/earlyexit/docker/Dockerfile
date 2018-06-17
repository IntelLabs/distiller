FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04

# clean up and refresh apt-get index
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get clean autoclean && \
    apt-get autoremove -y && \
    apt-get install -y git python-virtualenv python-pip

# Create py3 virtualenv in order to use python3 for the project
RUN virtualenv -p python3 /venv
RUN chmod -R 777 /venv
ENV PATH="/venv/bin:$PATH"
RUN pip install pip --upgrade

# Install interactive tools
RUN pip install ipython ipdb jupyter

# Apt cleaning
RUN apt-get clean autoclean && \
    apt-get autoremove -y

# install python requirements
ADD /docker/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
