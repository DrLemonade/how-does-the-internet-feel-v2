FROM python:3.11.2-slim-buster

WORKDIR /usr/src/app

RUN pip install --upgrade pip
COPY ./requirements.txt .
RUN pip install -r requirements.txt

# copy project
COPY . .