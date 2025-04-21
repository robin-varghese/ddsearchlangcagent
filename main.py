import os, json, logging, time
from typing import Dict

import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain.agents import AgentExecutor, Tool, create_openai_functions_agent
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.utils.function_calling import convert_to_openai_function

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import os
from fastapi import FastAPI
from google.cloud import secretmanager

# ... other imports ...

app = FastAPI()

def access_secret_version(secret_id, version_id="latest"):
    """Access the secret version."""
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{os.environ.get('GOOGLE_CLOUD_PROJECT')}/secrets/{secret_id}/versions/{version_id}"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")

# Get the API key from Secret Manager
try:
    os.environ["GOOGLE_CLOUD_PROJECT"]
except KeyError:
    raise ValueError("GOOGLE_CLOUD_PROJECT environment variable must be set.")

google_api_key = access_secret_version("google-api-key")

if not google_api_key:
    raise ValueError("Failed to retrieve GOOGLE_API_KEY from Secret Manager")

# ... Initialize your LangChain agent using google_api_key ...


logging.info("Configuring Google Generative AI with the provided API Key")
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize the FastAPI app
logging.info("Initializing FastAPI application")
app = FastAPI()

# Initialize the LLM
logging.info("Initializing ChatGoogleGenerativeAI LLM")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY)

# Initialize the DuckDuckGo search tool
logging.info("Initializing DuckDuckGo search tool")
search = DuckDuckGoSearchAPIWrapper()

tools = [
    Tool.from_function(func=search.run,name="Search", description="useful for when you need to answer questions from the user")
]

agent_prompt = ChatPromptTemplate.from_messages([MessagesPlaceholder(variable_name="chat_history"), HumanMessage(content="answer the user question using the available tools")])
logging.info("Creating LangChain agent")
agent = create_openai_functions_agent(llm, tools, agent_prompt)
logging.info("Creating LangChain agent executor")
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

@app.post("/search")
async def search_api(query: str) -> Dict[str, str]:
    logging.info(f"API request received: /search?query={query}")
    start_time = time.time()
    try:
        logging.info(f"Invoking LangChain agent_executor with query: {query}")
        response = agent_executor.invoke({"input": query})
        end_time = time.time()
        duration = end_time - start_time
        logging.info(f"LangChain agent_executor completed. Output: {response['output']}. Time taken: {duration:.4f} seconds")
        return {"result": response["output"]}
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        logging.error(f"Error processing API request: {e}. Time taken: {duration:.4f} seconds", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")
