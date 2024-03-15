import os
import pandas as pd
import pyodbc
from langchain_openai import AzureChatOpenAI
from langchain_community.agent_toolkits import create_sql_agent, SQLDatabaseToolkit
from langchain_community.utilities.sql_database import SQLDatabase
from langchain.memory import ConversationBufferWindowMemory
from sqlalchemy.engine.url import URL 

from langchain.agents import AgentExecutor
from common.prompts import MSSQL_AGENT_PREFIX, MSSQL_AGENT_SUFFIX, MSSQL_AGENT_FORMAT_INSTRUCTIONS

from IPython.display import Markdown, display  

from dotenv import load_dotenv
load_dotenv("credentials.env")

def printmd(string):
    display(Markdown(string))
    

class SQLSearchAgent:
    """Class for a ChatGPT with sql"""

    name = "@sql_chatgpt"
    description = "Ask questions directly to a SQL Database."
    
    def __init__(self):
        self.k = 100  
        self.llm = AzureChatOpenAI(  
            openai_api_key=os.environ['AZURE_OPENAI_API_KEY'],  
            api_version=os.environ['AZURE_OPENAI_API_VERSION'],  
            azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],  
            deployment_name=os.environ['AZURE_OPENAI_MODEL_NAME'],  
            model_name=os.environ['AZURE_OPENAI_MODEL_NAME'],  
            temperature=0  
        )  
  
        
        self.memory = ConversationBufferWindowMemory(memory_key="conversation_memory", return_messages=True, k=5)

    def _run(self, query: str) -> str:  
         # Configuration for the database connection
        db_config = {
            'drivername': 'mssql+pyodbc',
            'username': os.environ["SQL_SERVER_USERNAME"] + '@' + os.environ["SQL_SERVER_NAME"],
            'password': os.environ["SQL_SERVER_PASSWORD"],
            'host': os.environ["SQL_SERVER_NAME"],
            'port': 1433,
            'database': os.environ["SQL_SERVER_DATABASE"],
            'query': {'driver': 'ODBC Driver 17 for SQL Server'},
            }
        
            # Create a URL object for connecting to the database
        db_url = URL.create(**db_config)
                   
        db = SQLDatabase.from_uri(db_url)  
        toolkit = SQLDatabaseToolkit(db=db, llm=self.llm)

        agent_executor = create_sql_agent(
            prefix=MSSQL_AGENT_PREFIX,
            suffix=MSSQL_AGENT_SUFFIX,
            format_instructions = MSSQL_AGENT_FORMAT_INSTRUCTIONS,
            llm=self.llm,
            toolkit=toolkit,
            top_k=30,
            agent_type="openai-tools",
            verbose=True
        ) 

        try:  
            response = agent_executor.run(query)  
        except Exception as e:  
            response = str(e)  
            if 'Could not parse LLM output' in response:  
                response = "Sorry, I couldn't understand your query. Please try again with a different query.Please Note I can only provide questions related to the SQL Database i have access to"  
            else:  
                response = "Sorry, I encountered an error. Please try again later."  
        return response  

