FROM nvidia/cuda:12.0.1-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo
ENV SHELL=/bin/bash

ARG USERNAME=${USERNAME}
ARG USER_UID=${USER_UID}
ARG USER_GID=${USER_GID}

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10-dev \
    python3-pip \
    python3-setuptools

RUN groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

COPY ./requirements.txt /tmp/
RUN python3 -m pip install -U pip && \
    python3 -m pip --disable-pip-version-check --no-cache-dir install -r /tmp/requirements.txt

USER $USERNAME
WORKDIR /workspace
