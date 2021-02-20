
FROM python:3.8.8-slim

MAINTAINER MadMedian

RUN apt-get update

RUN mkdir batch-model
COPY requirements.txt /batch-model/

RUN pip install -r /batch-model/requirements.txt

ENV PATH="/batch-model/src:${PATH}"

COPY src /batch-model/src
WORKDIR /batch-model/src