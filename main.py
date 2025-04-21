import os, json, logging, time
from typing import Dict, List,Any

import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain.agents import AgentExecutor, Tool
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
#from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.agents import create_tool_calling_agent
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import os
from fastapi import FastAPI
from google.cloud import secretmanager

# ... other imports ...

app = FastAPI(title="LangChain Agent Search API")

def access_secret_version(secret_id, GOOGLE_CLOUD_PROJECT_NUMBER, version_id="latest"):
    """Access the secret version."""
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{GOOGLE_CLOUD_PROJECT_NUMBER}/secrets/{secret_id}/versions/{version_id}"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")

# Get the API key from Secret Manager
try:
    GOOGLE_CLOUD_PROJECT_NUMBER = os.environ.get("GOOGLE_CLOUD_PROJECT_NUMBER")
except KeyError:
    raise ValueError("GOOGLE_CLOUD_PROJECT_NUMBER environment variable must be set.")

GOOGLE_API_KEY = access_secret_version("google-api-key", GOOGLE_CLOUD_PROJECT_NUMBER)

if not GOOGLE_API_KEY:
    raise ValueError("Failed to retrieve GOOGLE_API_KEY from Secret Manager")

# ... Initialize your LangChain agent using google_api_key ...


logging.info("Configuring Google Generative AI with the provided API Key")
#TODO can be removed
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize the FastAPI app
logging.info("Initializing FastAPI application")


# Initialize the LLM
logging.info("Initializing ChatGoogleGenerativeAI LLM")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", 
                                temperature=0,
                                max_tokens=None,
                                timeout=None,
                                max_retries=2,
                                google_api_key=GOOGLE_API_KEY)

# Initialize the DuckDuckGo search tool
logging.info("Initializing DuckDuckGo search tool")
search = DuckDuckGoSearchAPIWrapper()

# Define tools with specific descriptions
tools = [
    # Tool class is often preferred now, but from_function still works
    # from langchain_core.tools import Tool <- Recommended import path
    Tool.from_function(
        func=search.run,
        name="DuckDuckGoSearch", # Specific name
        description="Useful for searching the web with DuckDuckGo to find current, real-time information about events, topics, or specific facts not found in the LLM's internal knowledge." # Specific description
    )
]

# Define the prompt for the tool-calling agent
# This prompt structure is generally suitable for tool calling agents
agent_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Answer the user's questions using the available tools."),
        MessagesPlaceholder(variable_name="chat_history", optional=True), # Make history optional if not always provided
        ("human", "{input}"), # User input
        MessagesPlaceholder(variable_name="agent_scratchpad"), # Place for intermediate agent steps
    ]
)

# Create the correct agent for Google tool calling
try:
    logging.info("Creating LangChain tool-calling agent")
    agent = create_tool_calling_agent(llm, tools, agent_prompt)
except Exception as e:
    logging.error(f"Failed to create agent: {e}", exc_info=True)
    raise

# Create the agent executor
logging.info("Creating LangChain agent executor")
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True # Good practice
)

# --- API Endpoint Definition ---

class SearchRequest(BaseModel):
    query: str
    # Optional: Include chat_history if your agent/prompt uses it
    chat_history: List[Dict[str, str]] = [] # Example: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]

class SearchResponse(BaseModel):  # Define SearchResponse here
    result: str

# Helper to format chat history for LangChain if needed
def format_chat_history(history: List[Dict[str, str]]) -> List:
    lc_history = []
    for msg in history:
        if msg.get("role") == "user":
            lc_history.append(HumanMessage(content=msg.get("content", "")))
        elif msg.get("role") == "assistant" or msg.get("role") == "ai":
             lc_history.append(AIMessage(content=msg.get("content", "")))
    return lc_history


@app.post("/search", response_model=SearchResponse)
async def search_api(request: SearchRequest) -> SearchResponse:
    logging.info(f"API request received: /search - Query: {request.query}")
    start_time = time.time()
    try:
        logging.info(f"Invoking LangChain agent_executor asynchronously with query: {request.query}")

        # Format history if provided
        formatted_history = format_chat_history(request.chat_history)

        # Use ainvoke and pass input correctly
        response = await agent_executor.ainvoke({
            "input": request.query,
            "chat_history": formatted_history # Pass formatted history
        })

        output = response.get("output", "No output generated.")
        end_time = time.time()
        duration = end_time - start_time
        logging.info(f"LangChain agent_executor completed. Output: {output}. Time taken: {duration:.4f} seconds")
        return SearchResponse(result=output)

    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        logging.error(f"Error processing API request: {e}. Time taken: {duration:.4f} seconds", exc_info=True)
        # Provide a generic error message to the client
        raise HTTPException(status_code=500, detail="An internal server error occurred while processing your request.")

# Add a root endpoint for health check
@app.get("/")
def read_root():
    return {"status": "API is running"}

# To run (save as main.py): uvicorn main:app --reload