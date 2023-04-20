FROM bitnami/pytorch:2.0.0-debian-11-r5

USER root
WORKDIR /root
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PYTHONPATH /root

RUN apt-get update && apt-get install -y build-essential libssl-dev curl ffmpeg

ENV VENV /opt/venv
RUN python3 -m venv ${VENV}
ENV PATH="${VENV}/bin:$PATH"

RUN pip install --no-cache-dir -U pip setuptools wheel

COPY requirements.txt /tmp
RUN pip install -r /tmp/requirements.txt

COPY sadtalker /root/sadtalker
