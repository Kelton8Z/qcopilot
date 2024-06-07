import os
import requests
import json
# from llama_index.readers.download import download_loader
# from llama_index.core import download_loader
import lark_oapi as lark
from lark_oapi.api.wiki.v2 import *
from lark_oapi.api.docx.v1 import *
from lark_oapi.api.auth.v3 import *
import streamlit as st
import openai
import anthropic
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.llms.anthropic import Anthropic
from llama_index.embeddings.jinaai import JinaEmbedding
from llama_index.core import Settings

try:
  from llama_index import VectorStoreIndex, ServiceContext, Document, SimpleDirectoryReader
except ImportError:
  from llama_index.core import VectorStoreIndex, ServiceContext, Document, SimpleDirectoryReader

from readFeishuWiki import readWiki
app_id = st.secrets.feishu_app_id
app_secret = st.secrets.feishu_app_secret
space_id = st.secrets.feishu_space_id
client = lark.Client.builder() \
        .enable_set_token(True) \
        .log_level(lark.LogLevel.DEBUG) \
        .app_id(app_id) \
        .app_secret(app_secret) \
        .build()

'''
def readWiki(space_id, app_id, app_secret):
    tenant_access_token = getTenantAccessToken(app_id, app_secret)

    # read doc
    client = lark.Client.builder() \
        .enable_set_token(True) \
        .log_level(lark.LogLevel.DEBUG) \
        .build()

    # 构造请求对象
    request: GetDocumentRequest = GetDocumentRequest.builder() \
        .document_id(doc_id) \
        .build()

    # 发起请求
    option = lark.RequestOption.builder().tenant_access_token(tenant_access_token).build()
    response: GetDocumentResponse = client.docx.v1.document.get(request, option)

    # 处理失败返回
    if not response.success():
        lark.logger.error(
            f"client.docx.v1.document.get failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}")
        return

    # 处理业务结果
    lark.logger.info(lark.JSON.marshal(response.data, indent=4))

    title = response.data.document.title

    request: RawContentDocumentRequest = RawContentDocumentRequest.builder() \
        .document_id(doc_id) \
        .lang(0) \
        .build()

    # 发起请求
    response: RawContentDocumentResponse = client.docx.v1.document.raw_content(request, option)
    with open("./data/"+title, 'w') as f:
        f.write(response.data.content)
'''

title = "AI assistant, powered by Qingcheng knowledge"
prompt = "You are an expert AI engineer in our company Qingcheng and your job is to answer technical questions. Keep your answers technical and based on facts – do not hallucinate features."
st.set_page_config(page_title=title, page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
openai.api_key = st.secrets.openai_key

os.environ["ANTHROPIC_API_KEY"] = st.secrets.claude_key
os.environ["JINAAI_API_KEY"] = st.secrets.jinaai_key

llm_map = {"claude": Anthropic(model="claude-3-opus-20240229"), 
           "gpt3.5": OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt=prompt),
           "ollama": Ollama(model="llama2", request_timeout=60.0)
}

st.title(title)
         
if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question!"}
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the docs – hang tight! This should take 1-2 minutes."):
        app_id = st.secrets.feishu_app_id
        app_secret = st.secrets.feishu_app_secret
        readWiki(space_id, app_id, app_secret)
        directory = "./data"
        if not os.path.exists(directory):
            os.makedirs(directory)
        reader = SimpleDirectoryReader(input_dir=directory, recursive=True)
        docs = reader.load_data()
        Settings.llm = llm_map["ollama"]
        embed_model = JinaEmbedding(
            api_key=st.secrets.jinaai_key,
            model="jina-embeddings-v2-base-en",
            embed_batch_size=16,
        )
        index = VectorStoreIndex.from_documents(docs, embed_model=embed_model)
        return index

index = load_data()

if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
    st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

prompt = st.chat_input("Your question")
if prompt: # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history


