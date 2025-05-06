import streamlit as st
import os
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.yfinance import YFinanceTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.models.google import Gemini


# Load environment variables
load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Streamlit Page Config
st.set_page_config(
    page_title="Finance & Web AI",
    page_icon="📈",
    layout="wide"
)

# --- Header ---
st.title("📈 AI-Powered Finance & Web Search Assistant")
st.markdown("🚀 Get real-time stock insights, analyst recommendations, the latest news, and chat with our finance AI assistant.")

# --- Define AI Agents ---

## Web Search Agent
web_search_agent = Agent(
    name="Web Search Agent",
    role="Search the web for the latest information",
    model=Gemini(id="gemini-2.0-flash"),
    tools=[DuckDuckGoTools()],
    instructions=[
        "ALWAYS present information in tabular format where possible",
        "Always include sources with dates of publication",
        "Structure your output with clear headings and bullet points",
        "For financial news, categorize information by market impact (Positive/Neutral/Negative) in a table format",
        "Include a summary of key takeaways at the end in a table format"
    ],
    show_tool_calls=True,
    markdown=True,
)

## Financial Agent
finance_agent = Agent(
    name="Finance AI Agent",
    model=Gemini(id="gemini-2.0-flash"),
    tools=[
        YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_news=True),
    ],
    instructions=[
        "ALWAYS present ALL data in tabular format - no exceptions",
        "Present analyst recommendations with consensus ratings in a table (Strong Buy/Buy/Hold/Sell/Strong Sell)",
        "Include target price ranges and average price targets in a dedicated table",
        "Provide technical indicators with clear buy/sell signals in a table format",
        "Format all price data with appropriate currency symbols",
        "Present 'Timing Guidance' section with short-term, medium-term, and long-term outlooks in a table",
        "Add a 'Risk Assessment' section in tabular format highlighting potential downsides",
        "Structure output with clear headings: Summary, Price Data, Fundamentals, Analyst Views, Technical Analysis, Timing Guidance, Risk Assessment",
        "Even summary information must be presented in a table format"
    ],
    show_tool_calls=True,
    markdown=True,
)

## Chatbot Agent with Web Search Capability
chatbot_agent = Agent(
    name="Finance Chatbot Agent",
    model=Groq(id="meta-llama/llama-4-maverick-17b-128e-instruct"),
    tools=[DuckDuckGoTools()],
    instructions=[
        "ALWAYS organize and present data in tables wherever possible",
        "Provide clear and concise answers to finance-related questions",
        "Include sources when fetching real-time data",
        "Structure responses with headings and tables - avoid bullet points where tables can be used instead",
        "For investment advice, always include timing considerations and risk factors in tabular format",
        "Use emojis sparingly to highlight important points",
        "Include a 'Key Takeaways' section at the end of comprehensive responses in a table format"
    ],
    show_tool_calls=True,
    markdown=True,
)

multi_chatbot_agent = Agent(
    team=[chatbot_agent, web_search_agent],
    model=Groq(id="meta-llama/llama-4-maverick-17b-128e-instruct"),
    instructions=[
        "ALWAYS present information in tables - consider this a strict requirement",
        "Always include sources with dates in tabular format",
        "Structure output with clear headings and tables for each section",
        "First use Finance Chatbot Agent to answer the question",
        "Then use the Web Search Agent for recent news",
        "Present a unified analysis that combines fundamental data with recent news in tabular format",
        "Include timing guidance (when to buy/sell) based on technical indicators and news sentiment in a table",
        "Clearly separate fact-based information from AI-generated analysis using separate tables",
        "End with actionable insights and key takeaways presented in a table"
    ],
    show_tool_calls=True,
    markdown=True,
)


# Combined Multi-Agent System
multi_ai_agent = Agent(
    team=[finance_agent, web_search_agent],
    model=Groq(id="meta-llama/llama-4-maverick-17b-128e-instruct"),
    instructions=[
        "ALWAYS present ALL information in table format - this is mandatory",
        "Structure output with clear sections using markdown headings, with each section containing at least one table",
        "First use the Finance Agent to get detailed stock data",
        "Then use the Web Search Agent for recent news and market sentiment",
        "Present ALL data in tables - never use paragraphs where tables can be used instead",
        "Include a 'Stock Fundamentals' table with key metrics and comparisons to industry averages",
        "Provide 'Analyst Consensus' table with specific ratings, target prices and timeframes",
        "Add 'Technical Analysis' table with key indicators and clear buy/sell signals",
        "Include 'Entry Points' table suggesting optimal buying opportunities based on technical patterns",
        "Add 'Investment Timeframe' table (Short-term trader vs. Long-term investor recommendations)",
        "Include 'Risk Assessment' table highlighting potential downside scenarios",
        "End with 'Action Plan' table summarizing recommendations with clear timing guidance",
        "Always cite sources for all external information in a dedicated sources table"
    ],
    show_tool_calls=True,
    markdown=True,
)

# --- Streamlit UI ---
st.sidebar.header("🔍 Search Options")
query_type = st.sidebar.selectbox(
    "Select Analysis Type",
    ["Stock Insights & Analyst Recommendations", "Latest News", "Talk to Chatbot"],
)

if query_type == "Talk to Chatbot":
    st.subheader("💬 Chat with Finance AI Assistant")
    user_question = st.text_input("Ask a finance-related question:")
    if st.button("Send"):
        with st.spinner("The AI Assistant is formulating a response..."):
            try:
                response = multi_chatbot_agent.run(user_question)
                st.markdown(response.content)
            except Exception as e:
                st.error(f"⚠️ Error: {e}")
else:
    stock_symbol = st.sidebar.text_input("Enter Stock Ticker (e.g., NVDA, AAPL)", value="NVDA")
    if st.sidebar.button("🚀 Get Insights"):
        with st.spinner("Fetching AI-powered insights... ⏳"):
            try:
                if query_type == "Stock Insights & Analyst Recommendations":
                    prompt = f"Summarize analyst recommendations and share the latest stock insights for {stock_symbol}."
                else:
                    prompt = f"Find the latest news about {stock_symbol}."

                response = multi_ai_agent.run(prompt)

                # Display results
                st.subheader("📝 AI-Generated Insights")
                st.markdown(response.content)

            except Exception as e:
                st.error(f"⚠️ Error fetching insights: {e}")

# --- Footer ---
st.markdown("---")
st.markdown("🔗 Powered by AI Agents, Groq and Gemini LLMs")
