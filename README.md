# Flask API Service Starter

This is a minimal Flask API service starter based on [Google Cloud Run Quickstart](https://cloud.google.com/run/docs/quickstarts/build-and-deploy/deploy-python-service).

## Getting Started

Server should run automatically when starting a workspace. To run manually, run:
```sh
./devserver.sh
```
gcloud secrets create google-api-key --replication-policy automatic
echo "replace with actual key" | gcloud secrets versions add google-api-key --data-file=-