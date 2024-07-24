import streamlit as st
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core.chat_engine import SimpleChatEngine

azure_api_key = st.secrets.azure_api_key
azure_endpoint = st.secrets.azure_endpoint
azure_chat_deployment = st.secrets.azure_chat_deployment
azure_embedding_deployment = st.secrets.azure_embedding_deployment

api_version = "2024-02-01"

llm = AzureOpenAI(
    model="gpt-35-turbo-16k",
    deployment_name=st.secrets.azure_chat_deployment,
    api_key=azure_api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
)
chat_engine = SimpleChatEngine.from_defaults(llm=llm)
response = chat_engine.chat("hi")
print(response.response)