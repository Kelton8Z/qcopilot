import os
import asyncio
import lark_oapi as lark
from lark_oapi.api.wiki.v2 import *
from lark_oapi.api.docx.v1 import *
from lark_oapi.api.auth.v3 import *
import streamlit as st
from llama_index.llms.ollama import Ollama

from llama_index.llms.openai import OpenAI
from llama_index.llms.anthropic import Anthropic
from llama_index.core import Settings

from readFeishuWiki import readWiki

from streamlit_feedback import streamlit_feedback
from langchain.callbacks.tracers.run_collector import RunCollectorCallbackHandler
from langchain.memory import StreamlitChatMessageHistory, ConversationBufferMemory
from langchain.schema.runnable import RunnableConfig
from langsmith import Client, traceable
from langchain.callbacks.tracers.langchain import wait_for_all_tracers
from langsmith.wrappers import wrap_openai
import cohere 

title = "AI assistant, powered by Qingcheng knowledge"
st.set_page_config(page_title=title, page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)

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

os.environ["JINAAI_API_KEY"] = st.secrets.jinaai_key
# os.environ["OPENAI_API_BASE"] = 
# os.environ["LANGCHAIN_ENDPOINT"] = langchain_endpoint ="https://api.moonshot.cn/v1"
os.environ["OPENAI_API_KEY"] = st.secrets.openai_key
os.environ["LANGCHAIN_TRACING_V2"] = "true" 
langchain_api_key = os.environ["LANGCHAIN_API_KEY"] = st.secrets.langsmith_key
    
langsmith_client = Client(api_key=langchain_api_key) #api_url=langchain_endpoint, 

llm_map = {"claude": Anthropic(model="claude-3-opus-20240229"), 
           "gpt4o": OpenAI(model="gpt-4o", system_prompt=prompt),
           "gpt3.5": OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt=prompt),
           "kimi": OpenAI(api_key = st.secrets.kimi_key, base_url = "https://api.moonshot.cn/v1"),
           "cohere": cohere.Client(api_key=st.secrets.cohere_key),
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
        from llama_index.embeddings.jinaai import JinaEmbedding
        embed_model = JinaEmbedding(
            api_key=st.secrets.jinaai_key,
            model="jina-embeddings-v2-base-en",
            embed_batch_size=16,
        )
        index = asyncio.run(readWiki(space_id, app_id, app_secret, embed_model))
        
        return index
    
def submit_feedback(e=None):
    if e:
        raise e
    else:
        setattr(st.session_state, 'disable_chat', False)
        st.rerun()
        # st.experimental_rerun()
    
@traceable
def main():
    # from authFeishu import Auth
    # auth = Auth("https://open.feishu.cn", app_id, app_secret)
    # auth.authorize_app_access_token()

    Settings.llm = llm_map["gpt3.5"]
    index = load_data()
    memory = ConversationBufferMemory(
        chat_memory=StreamlitChatMessageHistory(key="langchain_messages"),
        return_messages=True,
        memory_key="chat_history",
    )
    if st.sidebar.button("Clear message history"):
        print("Clearing message history")
        memory.clear()
        st.session_state.trace_link = None
        st.session_state.run_id = None
        
    last_run = next(langsmith_client.list_runs(
        project_name="default"
    ))

    if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
        st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)
    if 'disable_chat' not in st.session_state:
        print("dne\n")
        st.session_state.disable_chat = False

    prompt = st.chat_input("Your question", disabled=st.session_state.disable_chat)
    if prompt: # Prompt for user input and save to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # st.session_state.disable_chat = True # Disable input until the assistant responds

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
                st.session_state.disable_chat = True # Disable message input until feedback is submitted
                feedback_option = "faces" if st.toggle(label="`Thumbs` ⇄ `Faces`", value=True) else "thumbs"
                feedback = streamlit_feedback(
                    feedback_type=feedback_option,  # Apply the selected feedback style
                    optional_text_label="[Optional] Please provide an explanation",  # Allow for additional comments
                    on_submit=submit_feedback,
                    key=f"feedback_{last_run.id}",
                )
                
                # assert st.session_state.disable_chat, "Feedback submission did not update voting state"


main()
