# Stage 1: Build the dependencies
FROM python:3.11-slim-buster as builder
WORKDIR /app
COPY requirements.txt .

# Create and use a virtual environment in the builder stage
RUN python3 -m venv .venv && \
    . .venv/bin/activate && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Create the final image
FROM python:3.11-slim-buster
ENV PYTHONUNBUFFERED True
WORKDIR /app

# Copy the virtual environment
COPY --from=builder /app/.venv .venv  
COPY . .

# Use gunicorn from the copied virtual environment
CMD exec /app/.venv/bin/gunicorn --bind :$PORT --workers 1 --threads 8 main:app



