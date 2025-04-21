# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.11-slim-buster

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

# Copy local code to the container image.
WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt
# Stage 1: Build the dependencies
COPY . .
FROM python:3.11-slim-buster as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Create the final image
FROM python:3.11-slim-buster
ENV PYTHONUNBUFFERED True
WORKDIR /app
# Copy only the installed dependencies
COPY --from=builder /app .  
# Copy the rest of your application code
COPY . .  
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 main:app


