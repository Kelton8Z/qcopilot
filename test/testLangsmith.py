import os 
import streamlit as st
from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_BASE"] = "http://vasi.chitu.ai/v1"
os.environ["OPENAI_API_KEY"] = st.secrets.openai_key

os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"]=st.secrets.langsmith_key
os.environ["LANGCHAIN_PROJECT"]="July"

llm = ChatOpenAI()
llm.invoke("Hello, world!")