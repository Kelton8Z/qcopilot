import os
import streamlit as st
import chromadb
import tiktoken
from llama_index.llms.openai import OpenAI
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core import Settings, StorageContext, VectorStoreIndex, SimpleDirectoryReader

from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding

os.environ["OPENAI_API_BASE"] = st.secrets.openai_api_base
os.environ["OPENAI_API_KEY"] = st.secrets.openai_key 
collection = "qcWiki"
chroma_db_path = "chroma_db"
embed_model = OpenAIEmbedding(model="text-embedding-3-large", api_base=openai_api_base)

documents = SimpleDirectoryReader("./data").load_data()

db = chromadb.PersistentClient(path=chroma_db_path)
chroma_collection = db.get_or_create_collection(collection)
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

token_counter = TokenCountingHandler(
    tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode
)

Settings.callback_manager = CallbackManager([token_counter])
Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.2)
'''
index = VectorStoreIndex.from_vector_store(
    vector_store,
    embed_model=embed_model,
)'''
index = VectorStoreIndex.from_documents(documents)
print(token_counter)
print(f'{token_counter.total_embedding_token_count} tokens indexed!')
