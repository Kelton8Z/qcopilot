import streamlit as st
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
import chromadb

collection = "qcWiki"
chroma_db_path = "chroma_db"
fileToTitleAndUrl = {}

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

db = chromadb.PersistentClient(path=chroma_db_path)
chroma_collection = db.get_collection(collection)
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_vector_store(
    vector_store,
    storage_context=storage_context,
    embed_model=embed_model,
    store_nodes_override=True
)

query_engine = index.as_query_engine()
model = "llama3"
input_len = 60
output_len = 20
ch_question = f'{model}模型输入长度为{input_len}输出长度为{output_len}时吞吐量(request/s)，吞吐量(tokens/s)，batch是1、8、64的平均延迟分别是多少? 只返回最终答案数字[Throughput（requests/s), Throughput（tokens/s), Avg latency（Batch=1）, Avg latency（Batch=8）, Avg latency（Batch=64）]'
print(query_engine.query(ch_question))