import os
import asyncio
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
from llama_index.core import Settings

from readFeishuWiki import readWiki
title = "AI assistant, powered by Qingcheng knowledge"
st.set_page_config(page_title=title, page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)

os.environ["OPENAI_API_BASE"] = "https://vasi.chitu.ai/v1"
os.environ["OPENAI_API_KEY"] = st.secrets.openai_key
app_id = st.secrets.feishu_app_id
app_secret = st.secrets.feishu_app_secret
space_id = st.secrets.feishu_space_id
client = lark.Client.builder() \
        .enable_set_token(True) \
        .log_level(lark.LogLevel.DEBUG) \
        .app_id(app_id) \
        .app_secret(app_secret) \
        .build()

prompt = "You are an expert AI engineer in our company Qingcheng and your job is to answer technical questions. Keep your answers technical and based on facts – do not hallucinate features."
openai.api_key = st.secrets.openai_key

os.environ["ANTHROPIC_API_KEY"] = st.secrets.claude_key
os.environ["JINAAI_API_KEY"] = st.secrets.jinaai_key

llm_map = {"claude": Anthropic(model="claude-3-opus-20240229"), 
           "gpt4o": OpenAI(model="gpt-4o", system_prompt=prompt),
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
        # recursively read wiki and write each file into the machine
        index = asyncio.run(readWiki(space_id, app_id, app_secret))
        
        return index
    
def main():
    # from authFeishu import Auth
    # auth = Auth("https://open.feishu.cn", app_id, app_secret)
    # auth.authorize_app_access_token()

    Settings.llm = llm_map["gpt4o"]
    index = load_data()

    if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
        st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)
    if 'voted' not in st.session_state:
        st.session_state.voted = False

    prompt = st.chat_input("Your question", disabled=not st.session_state.voted)
    if prompt: # Prompt for user input and save to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
    if 'thumbs_up' not in st.session_state:
        st.session_state.thumbs_up = 0

    if 'thumbs_down' not in st.session_state:
        st.session_state.thumbs_down = 0

    for message in st.session_state.messages: # Display the prior chat messages
        with st.chat_message(message["role"]):
            st.write(message["content"])

    @st.experimental_dialog("Thumb up/down")
    def vote(item):
        st.session_state.voted = True
        st.write(f"Why {item}?")
        reason = st.text_input("Because...")
        if st.button("Submit"):
            st.session_state.vote = {"item": item, "reason": reason}
            st.rerun()

    # If last message is not from assistant, generate a new response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chat_engine.chat(prompt)
                st.write(response.response)
                message = {"role": "assistant", "content": response.response}
                st.session_state.messages.append(message) # Add response to message history
        st.session_state.voted = False
    else:
        if not st.session_state.voted:
            if st.button("👍"):
                vote("👍")
            if st.button("👎"):
                vote("👎")
main()
