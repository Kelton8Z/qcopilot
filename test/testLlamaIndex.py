import os
import streamlit as st
from llama_index.core import Settings
from llama_index.core import Document
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex

from llama_index.llms.anthropic import Anthropic
anthropic_api_base = "https://vasi.chitu.ai/v1/messages"
os.environ["ANTHROPIC_API_KEY"] = st.secrets.claude_key
os.environ["ANTHROPIC_API_BASE"] = anthropic_api_base
    
Settings.llm = Anthropic(model="claude-3-5-sonnet-20240620", base_url=anthropic_api_base) #OpenAI(model="gpt-3.5-turbo")

index = VectorStoreIndex.from_documents([Document(text="blah")])
st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", streaming=True)
prompt = "hello"
streaming_response = st.session_state.chat_engine.stream_chat(prompt)
print("".join([c for c in streaming_response.response_gen]))