import os
import sys
import glob
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub # Hub for prompt templates
from langchain_core.messages import HumanMessage, SystemMessage
from google.generativeai.types.safety_types import HarmBlockThreshold, HarmCategory
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

# --- Cleanup Function ---
# def cleanup_previous_files():
#     """
#     Removes files created in previous runs.
#     """
#     logger.info("Cleaning up previous run's files...")
#     try:
#         for filename in glob.glob("*.json") + glob.glob("*.bk"):
#             os.remove(filename)
#         logger.info("Cleanup successful.")
#     except Exception as e:
#         logger.error(f"Error during cleanup: {e}")


# --- Configuration ---
# Ensure the GOOGLE_API_KEY environment variable is set
if "GOOGLE_API_KEY" not in os.environ:
    logger.error("GOOGLE_API_KEY environment variable not set.")
    logger.error("Please set it before running the script:")
    logger.error("Linux/macOS: export GOOGLE_API_KEY='YOUR_API_KEY'")
    logger.error("Windows CMD: set GOOGLE_API_KEY=YOUR_API_KEY")
    logger.error("Windows PowerShell: $env:GOOGLE_API_KEY='YOUR_API_KEY'")
    sys.exit(1)
else:
    logger.info("GOOGLE_API_KEY environment variable found.")

# Configure safety settings for Gemini (Optional but recommended)
# Adjust these thresholds as needed (BLOCK_NONE allows most content)
safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE, 
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}

# --- Initialize Components ---

# 1. LLM: Google Gemini
logger.info("Initializing LLM (Google Gemini)...")
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.7, # Controls creativity (0=deterministic, 1=max creativity)
        convert_system_message_to_human=True, # Helps Gemini understand system prompts
        safety_settings=safety_settings
    )
    # Quick test to ensure LLM is working (optional)
    llm.invoke([HumanMessage(content="Hi")])
except Exception as e:
    logger.error(f"Error initializing LLM: {e}")
    logger.error("Check your API key and internet connection.")
    sys.exit(1)

# 2. Tool: DuckDuckGo Search
logger.info("Initializing Tools (DuckDuckGo Search)...")
search_tool = DuckDuckGoSearchRun()
tools = [search_tool]
logger.info("Tools Initialized Successfully.")

# 3. Agent Prompt Template
# We'll use a standard ReAct (Reasoning and Acting) prompt from Langchain Hub
logger.info("Fetching ReAct prompt template...")
try:
    # Pull the prompt template suitable for ReAct agents
    prompt = hub.pull("hwchase17/react")
    logger.info("Prompt template fetched successfully.")
    #logger.info("\n--- Prompt Template ---")
    # logger.info(prompt.template) # Uncomment to see the prompt structure
    # logger.info("----------------------\n")
except Exception as e:
    logger.error(f"Error fetching prompt from Langchain Hub: {e}")
    sys.exit(1)

# 4. Create the Agent
# The agent decides which tool to use based on the input and prompt
logger.info("Creating the ReAct agent...")
try:
    agent = create_react_agent(llm, tools, prompt)
    logger.info("Agent created successfully.")
except Exception as e:
    logger.error(f"Error creating agent: {e}")
    sys.exit(1)

# 5. Agent Executor
# This runs the agent's reasoning loop (Thought -> Action -> Observation -> Thought...)
logger.info("Creating Agent Executor...")
# verbose=True shows the agent's thought process, which is very helpful!
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True, # Gracefully handle if LLM output isn't perfect
    max_iterations=5 # Prevent infinite loops
)
logger.info("Agent Executor created successfully.")
logger.info("\n--- AI Search Agent Ready ---")


# --- FastAPI Setup ---
app = FastAPI()

# Perform cleanup when the server starts
#cleanup_previous_files()

# --- Request Model ---
class SearchRequest(BaseModel):
    query: str

class SearchResponse(BaseModel):
    response: str

@app.exception_handler(Exception)
async def validation_exception_handler(request: Request, err: Exception):
    base_error_message = f"Failed to execute: {request.method}: {request.url}"
    logger.error(f"{base_error_message}, error: {err}")
    return HTTPException(status_code=500, detail=f"{base_error_message}, error: {err}")

@app.post("/search", response_model=SearchResponse)
async def search(search_request: SearchRequest):
    """
    Handles search queries via the AI agent.
    """
    try:
        logger.info(f"Received search request: {search_request.query}")
        if not search_request.query:
             raise HTTPException(status_code=400, detail="Query cannot be empty.")

        logger.info("Agent thinking...")        
        response = agent_executor.invoke({"input": search_request.query})
        logger.info(f"Agent response: {response['output']}")
        return SearchResponse(response=response['output'])

    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

@app.get("/health")
async def health_check():
    """
    Performs a health check.
    """
    logger.info("Health check requested.")
    return {"status": "ok"}