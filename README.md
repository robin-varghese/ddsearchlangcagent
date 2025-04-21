# LangChain Agent Search API Service

This service provides an API powered by a LangChain agent that uses DuckDuckGo search to answer user questions. It is designed to be deployed on Google Cloud Run and leverages Google Cloud Secret Manager for storing the Google API key.

## Features

*   **LangChain Agent:** Utilizes a LangChain agent for question answering.
*   **DuckDuckGo Search:** Employs DuckDuckGo search to find real-time information.
*   **Google Cloud Secret Manager:** Securely stores the Google API key.
*   **FastAPI:** Uses FastAPI for building the API.
*   **Cloud Run Deployment:** Designed for deployment on Google Cloud Run.
*   **Chat History:** Supports optional chat history for context-aware conversations.
* **Health check:** provides a simple health check at `/` endpoint.

## API Endpoint

### `/search` (POST)

This endpoint accepts a JSON payload with the following structure:


```sh
./devserver.sh
```
gcloud secrets create google-api-key --replication-policy automatic
echo "replace with actual key" | gcloud secrets versions add google-api-key --data-file=-

gcloud run deploy ddsearchlangcagent --source . --project=vector-search-poc --region=us-central1 --set-env-vars GOOGLE_CLOUD_PROJECT=vector-search-poc,GOOGLE_CLOUD_PROJECT_NUMBER=xxxxx



curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the capital of France?", "chat_history": []}' \
  https://ddsearchlangcagent-xxxx.us-central1.run.app