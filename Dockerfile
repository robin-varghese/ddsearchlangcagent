# Use an official Python runtime as a parent image
# Using python:3.10-slim or python:3.11-slim is often a good balance
FROM python:3.11-slim

# Set environment variables
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    # Set the PORT environment variable Cloud Run expects (though we also specify in CMD)
    PORT=8080

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed system dependencies (if any - less common for basic FastAPI)
# RUN apt-get update && apt-get install -y --no-install-recommends some-package && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python dependencies from requirements.txt
# Using --no-cache-dir reduces image size
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container at /app
# Assumes your main.py and any other modules are in the same directory as the Dockerfile
COPY . .