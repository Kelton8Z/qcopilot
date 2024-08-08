import streamlit as st
from sentence_transformers import util
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

api_version="2024-02-01"
azure_api_key = st.secrets.azure_api_key
azure_endpoint = st.secrets.azure_endpoint
azure_chat_deployment = st.secrets.azure_chat_deployment
azure_embedding_deployment = st.secrets.azure_embedding_deployment

embed_model = AzureOpenAIEmbedding(
    model="text-embedding-ada-002",
    deployment_name=azure_embedding_deployment,
    api_key=azure_api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
)

# Generate embeddings for the keywords
embedding1 = embed_model.get_text_embedding('llama')
embedding2 = embed_model.get_text_embedding('llama3-7b')

# Calculate similarity
similarity = util.pytorch_cos_sim(embedding1, embedding2)

print(f"Similarity between 'llama3-7b' and 'llama': {similarity.item()}")