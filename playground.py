import openai 
from phi.agent import Agent, RunResponse
import phi.api
from phi.model.openai import OpenAIChat
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv 
from phi.model.groq import Groq 


import os 
import phi 
from phi.playground import Playground, serve_playground_app 

load_dotenv()   

phi.api = os.getenv("PHI_API_KEY")
# we will create a custom chatbot on the phidata platform 

web_search_agent = Agent(
    name = "Web Search Agent",
    role = "Search the web for information",
    model=Groq(id ="llama3-8b-8192"),
    tools = [DuckDuckGo()],
    instructions =["Always include source"],
    show_tools_calls = True,
    markdown = True, 
)

# Financial agent that can use the YFinance tools to get stock data
financial_agent = Agent(
    name = "Financial AI Agent",
    role = "Interact with financial data and provide insights",
    model = Groq(id="llama3-8b-8192"),
    tools = [
        YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_news=True),
    ],
    instructions = ["Use tables to display the data"],
    show_tools_calls = True,
    markdown = True,
)

app = Playground(agents=[financial_agent, web_search_agent]).get_app()

if __name__== "__main__":
    serve_playground_app("playground:app", reload=True)
    # This will start the playground app on http://localhost:5000
    # You can then interact with the agents through the web interface
