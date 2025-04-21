import os, json
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

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

genai.configure(api_key=GOOGLE_API_KEY)

app = FastAPI()

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY)
search = DuckDuckGoSearchAPIWrapper()

tools = [
    Tool.from_function(func=search.run,name="Search", description="useful for when you need to answer questions from the user")
]

agent_prompt = ChatPromptTemplate.from_messages([MessagesPlaceholder(variable_name="chat_history"), HumanMessage(content="answer the user question using the available tools")])
agent = create_openai_functions_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

@app.post("/search")
async def search_api(query: str) -> Dict[str, str]:
  response = agent_executor.invoke({"input": query})
  return {"result": response["output"]}