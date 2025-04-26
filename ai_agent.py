# Setup API keys for Groq, OpenAI, and Tavily
from dotenv import load_dotenv
import os

load_dotenv()

# Direct assignment of keys (as you requested)
GROQ_API_KEY = "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"
TAVILY_API_KEY = "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"
OPENAI_API_KEY = "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"

# Setup LLM & tools
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from langchain_core.messages.ai import AIMessage

# OpenAI error handling
import openai

system_prompt_default = "Act as an AI chatbot who is smart and friendly"

def get_response_from_ai_agent(llm_id, query, allow_search, system_prompt, provider):
    try:
        if provider == "Groq":
            llm = ChatGroq(model=llm_id, api_key=GROQ_API_KEY)
        elif provider == "OpenAI":
            llm = ChatOpenAI(model=llm_id, api_key=OPENAI_API_KEY)
        else:
            return "Invalid provider specified."

        tools = [TavilySearchResults(max_results=2, tavily_api_key=TAVILY_API_KEY)] if allow_search else []

        agent = create_react_agent(
            model=llm,
            tools=tools,
            state_modifier=system_prompt or system_prompt_default
        )

        state = {"messages": query}
        response = agent.invoke(state)
        messages = response.get("messages", [])
        ai_messages = [message.content for message in messages if isinstance(message, AIMessage)]

        if not ai_messages:
            return "No response from the AI agent."

        return ai_messages[-1]
    
    except openai.RateLimitError:
        return "⚠️ OpenAI API Rate Limit Exceeded. Please check your quota and billing at https://platform.openai.com/account/billing."
    
    except openai.AuthenticationError:
        return "❌ OpenAI API Authentication Failed. Please check your API key."
    
    except Exception as e:
        return f"🚨 Unexpected error occurred: {str(e)}"
