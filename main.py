import os
import time
import uuid
import random
import asyncio
import requests
import lark_oapi as lark
from lark_oapi.api.wiki.v2 import *
from lark_oapi.api.docx.v1 import *
from lark_oapi.api.auth.v3 import *
import streamlit as st
import openai
from multiprocessing import Process, Queue, set_start_method
from functools import partial
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.llms.anthropic import Anthropic
from llama_index.core import Settings, SimpleDirectoryReader
from llama_index.core.chat_engine import SimpleChatEngine

from llama_index.core.postprocessor import SimilarityPostprocessor

from readFeishuWiki import readWiki, ExcelReader, TsvReader
from S3ops import put_object, upload_file, create_bucket, create_presigned_url, delete_all_objects, delete_bucket, bucket_exists

from streamlit_feedback import streamlit_feedback
from langsmith.run_helpers import get_current_run_tree
from langchain_core.tracers.context import tracing_v2_enabled
#from langsmith import Client, traceable

title = "AI assistant, powered by Qingcheng knowledge"
st.set_page_config(page_title=title, page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)

openai_api_base = "http://vasi.chitu.ai/v1"
os.environ["OPENAI_API_BASE"] = openai_api_base
os.environ["OPENAI_API_KEY"] = st.secrets.openai_key


os.environ["AWS_ACCESS_KEY_ID"] = st.secrets.aws_access_key
os.environ["AWS_SECRET_ACCESS_KEY"] = st.secrets.aws_secret_key

os.environ["LANGCHAIN_PROJECT"] = "July"
os.environ["LANGCHAIN_TRACING_V2"] = "true" 
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
langchain_api_key = os.environ["LANGCHAIN_API_KEY"] = st.secrets.langsmith_key

langsmith_project_id = st.secrets.langsmith_project_id
#langsmith_client = Client(api_key=langchain_api_key)

app_id = st.secrets.feishu_app_id
app_secret = st.secrets.feishu_app_secret
space_id = st.secrets.feishu_space_id
client = lark.Client.builder() \
        .enable_set_token(True) \
        .log_level(lark.LogLevel.DEBUG) \
        .app_id(app_id) \
        .app_secret(app_secret) \
        .build()
        

# Initialize session state
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

if 'session_id' not in st.session_state or not st.session_state.session_id:
    st.session_state['session_id'] = str(uuid.uuid4())

prompt = "You are an expert ai infra analyst at 清程极智. Use your knowledge base to answer questions about ai model/hardware performance. Show URLs of your sources whenever possible"
openai.api_key = st.secrets.openai_key

os.environ["ANTHROPIC_API_KEY"] = st.secrets.claude_key
os.environ["JINAAI_API_KEY"] = st.secrets.jinaai_key

st.title(title)
_ = '''    
def _submit_feedback(user_response, emoji=None, run_id=None):
    feedback = user_response['score']
    feedback_text = user_response['text']
    # st.toast(f"Feedback submitted: {feedback}", icon=emoji)
    messages = st.session_state.messages
    if len(messages)>1:
        langsmith_client.create_feedback(
            run_id,
            key="user-score",
            score=0.0 if feedback=="👎" else 1.0,
            comment=f'{messages[-2]["content"]} + {messages[-1]["content"]} -> ' + feedback_text if feedback_text else "",
        )
    return user_response
'''

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the docs – hang tight! This should take 1-2 minutes."):
        app_id = st.secrets.feishu_app_id
        app_secret = st.secrets.feishu_app_secret
        # recursively read wiki and write each file into the machine
        # from llama_index.embeddings.jinaai import JinaEmbedding
        # embed_model = JinaEmbedding(
        #     api_key=st.secrets.jinaai_key,
        #     model="jina-embeddings-v2-base-en",
        #     embed_batch_size=16,
        # )
        from llama_index.embeddings.openai import OpenAIEmbedding
        embed_model = OpenAIEmbedding(model="text-embedding-3-large", api_base=openai_api_base)
        # from llama_index.core import VectorStoreIndex
        # index = VectorStoreIndex.from_documents([], embed_model=embed_model)
        index, fileToTitleAndUrl = asyncio.run(readWiki(space_id, app_id, app_secret, embed_model))
        
        return index, fileToTitleAndUrl 
   

llm_map = {"Claude3.5": Anthropic(model="claude-3-5-sonnet-20240620", system_prompt=prompt), 
           "gpt4o": OpenAI(model="gpt-4o", system_prompt=prompt),
           "gpt3.5": OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt=prompt),
           "Llama3_8B": OpenAI(base_url="http://localhost:25121/v1", system_prompt=prompt),
           "ollama": Ollama(model="llama2", request_timeout=60.0)
}

def toggle_llm():
    llm = st.sidebar.selectbox(
        "模型切换",
        ("gpt4o", "Claude3.5", "Llama3_8B")
    )
    os.environ["OPENAI_API_KEY"] = st.secrets.openai_key

    if llm!=st.session_state["llm"]:
        st.session_state["llm"] = llm
        Settings.llm = llm_map[llm]


def read_documents(directory, s3fs, queue, max_retries=5, retry_delay=2):
    for attempt in range(max_retries):
        try:
            reader = SimpleDirectoryReader(
                input_dir=directory,
                fs=s3fs,
                recursive=True,
                filename_as_id=True,
                file_extractor={".xlsx": ExcelReader(), ".tsv": TsvReader()},
                file_metadata=lambda filename: {"file_name": filename},
                raise_on_error=True
            )
            docs = reader.load_data()
            queue.put(docs)
            return  # Exit the function if successful
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"File not loaded, retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
            else:
                queue.put(None)
                print(f"Failed to load file after {max_retries} attempts: {e}")
                return

def toggle_rag_use():
    
    from llama_index.core import VectorStoreIndex
    from llama_index.core import Document
        
    use_rag = st.sidebar.selectbox(
        "是否用飞书知识库",
        ("否", "是")
    )
    use_rag = True if use_rag=="是" else False
   

    if use_rag!= st.session_state.use_rag:
        st.session_state.chat_engine = SimpleChatEngine.from_defaults(llm=llm_map[st.session_state["llm"]])
        if use_rag:
            index, fileToTitleAndUrl = load_data()                
            if index:
                st.session_state.fileToTitleAndUrl = fileToTitleAndUrl 
                st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", streaming=True)
            else:
                st.toast("调用飞书知识库失败")
                use_rag = False
        
        st.session_state.use_rag = use_rag
        st.rerun()
    else:
        if "chat_engine" not in st.session_state.keys():
            st.session_state.chat_engine = SimpleChatEngine.from_defaults(llm=llm_map[st.session_state["llm"]])
            if use_rag:
                index, fileToTitleAndUrl = load_data()                
                if index:
                    st.session_state.fileToTitleAndUrl = fileToTitleAndUrl 
                    st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", streaming=True)

    if not use_rag:
        uploaded_files = st.sidebar.file_uploader(label="上传临时文件", accept_multiple_files=True)
        if st.session_state.uploaded_files!=uploaded_files:
            old_files = [file.name for file in st.session_state.uploaded_files]
            cur_files = [file.name for file in uploaded_files]
            new_filenames = list(set(cur_files)-set(old_files))
            new_files = [file for file in uploaded_files if file.name in new_filenames] 
            st.session_state.uploaded_files = uploaded_files
            if uploaded_files:
                with st.status(label="上传处理中", expanded=True) as status:
                    st.write("上传云端")
                    use_rag = False

                    directory = st.session_state.session_id
                    if st.secrets.aws_region=='us-east-1':
                        region = None
                    else:
                        region = st.secrets.aws_region
                    
                    # handle uploading more files to the same bucket within the same session
                    success = False
                    bucket_created = False
                    is_bucket_exist = bucket_exists(directory)
                    if not is_bucket_exist:
                        bucket_created = create_bucket(bucket_name=directory, region=region)
                    if bucket_created or is_bucket_exist:
                        for file in new_files:
                            filename = file.name
                            filepath = directory+"/"+filename
                            bytes_data = file.read()
                            put_object(obj=bytes_data, bucket=directory, key=filename)
                            s3_url = create_presigned_url(bucket_name=directory, object_name=filename)
                            st.session_state.fileToTitleAndUrl[filepath] = {"title": filename, "url": s3_url}
                    
                        st.write("向量索引")    
                        
                        from s3fs import S3FileSystem
                        import boto3
                        endpoint = boto3.client("s3", region_name=st.secrets.aws_region).meta.endpoint_url
                        s3fs = S3FileSystem(anon=False, endpoint_url=endpoint)
                       
                        set_start_method('spawn', force=True)

                        max_retries = 5
                        retry_delay = 2  # seconds
                        
                        queue = Queue()

                        # Create and start the subprocess
                        process = Process(target=read_documents, args=(directory, s3fs, queue, max_retries, retry_delay))
                        process.start()

                        # Wait for the process to complete
                        process.join()

                        # Retrieve the result from the queue
                        docs = queue.get()

                        # Ensure the subprocess is terminated
                        if process.is_alive():
                            process.terminate()

                        if docs:
                            try:
                                index = VectorStoreIndex.from_documents(docs)
                                st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", streaming=True)
                                success = True
                            except:
                                success = False

                    if success:
                        status.update(label="上传完成", state="complete", expanded=False)
                    else:
                        status.update(label="上传失败", state="error", expanded=False)

def init_chat():
             
    if "llm" not in st.session_state.keys(): 
        st.session_state.llm = "gpt4o"
    if "use_rag" not in st.session_state.keys(): 
        st.session_state.use_rag = True
    if "fileToTitleAndUrl" not in st.session_state.keys(): 
        st.session_state.fileToTitleAndUrl = {}
    
    toggle_llm()
    toggle_rag_use()
    
    if "messages" not in st.session_state.keys() or len(st.session_state.messages)==0 or st.sidebar.button("清空对话"): # Initialize the chat messages history
        st.session_state.run_id = None
        st.session_state.chat_engine.reset()
        st.session_state.messages = []

def starter_prompts():
    prompt = ""
    demo_prompts = ["应该如何衡量decode和prefill过程的性能?", "AI Infra行业发展的目标是什么?", "JSX有什么优势?", "怎么实现capcha/防ai滑块?", "官网首页展示有哪些前端方案?", "我们的前端开发面试考察些什么?", "介绍一下CHT830项目", "llama模型平均吞吐量(token/s)多少?"]
    selected_prompts = random.sample(demo_prompts, 4)

    st.markdown("""<style>
    .stButton button {
        display: inline-block;
        width: 100%;
        height: 80px;
    }
    </style>""", unsafe_allow_html=True)
                
    cols = st.columns(4, vertical_alignment="center")
    for i, demo_prompt in enumerate(selected_prompts):
        with cols[i]:
            if st.button(demo_prompt):
                prompt = demo_prompt
                break

    return prompt

#@traceable(name=st.session_state.session_id)
def main():
    '''
    run = get_current_run_tree()
    run_id = str(run.id)
    st.session_state.run_id = st.session_state["run_0"] = run_id
    '''
    feedback_option = "faces" if st.toggle(label="`Thumbs` ⇄ `Faces`", value=False) else "thumbs"

    feedback_kwargs = {
        "feedback_type": feedback_option,
        "optional_text_label": "Please provide extra information",
    }
    
    init_chat()

    prompt = ""
    if len(st.session_state.messages)==0 and st.session_state.use_rag:
        prompt = starter_prompts() 
        # col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        # with col4:
        #     if st.button("试试别的问题"):
        #         selected_prompts = random.sample(demo_prompts, 4)

    if not prompt:
        # Prompt for user input and save to chat history
        prompt = st.chat_input("Your question")
    if prompt: 
        st.session_state.messages.append({"role": "user", "content": prompt})        

    # Display the prior chat messages
    for i, message in enumerate(st.session_state.messages): 
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
        _ = ''' 
        if message["role"]=="assistant":
            feedback_key = f"feedback_{int(i/2)}"
            # This actually commits the feedback
            streamlit_feedback(
                **feedback_kwargs,
                key=feedback_key,
                on_submit=partial(
                    _submit_feedback, run_id=st.session_state[f"run_{int(i/2)}"]
                ),
            )
        '''

    if st.session_state.messages:
        message = st.session_state.messages[-1]
        # If last message is not from assistant, generate a new response
        if message["role"] != "assistant":
            with st.chat_message("assistant"):
                response_container = st.empty()  # Container to hold the response as it streams
                response_msg = ""
                try:
                    if prompt:
                        streaming_response = st.session_state.chat_engine.stream_chat(prompt)
                    else:
                        st.rerun()
                except:
                    st.rerun()
                for token in streaming_response.response_gen:
                    response_msg += token
                    response_container.write(response_msg)
                
                if st.session_state.use_rag or st.session_state.uploaded_files:
                    processor = SimilarityPostprocessor(similarity_cutoff=0.25)
                    source_nodes = streaming_response.source_nodes
                    filtered_nodes = processor.postprocess_nodes(source_nodes)
                    sources_list = []
                    for node in source_nodes:
                        try:
                            if "file_path" in node.metadata:
                                file_path = node.metadata["file_path"]
                            else:
                                file_path = node.metadata["file_name"]
                            file_name = st.session_state.fileToTitleAndUrl[file_path]["title"]
                            file_url = st.session_state.fileToTitleAndUrl[file_path]["url"]
                            source = "[%s](%s)中某部分相似度" % (file_name, file_url) + format(node.score, ".2%") 
                            sources_list.append(source)
                        except Exception as e:
                            # no source wiki node
                            print(e)
                            pass
                    if sources_list: 
                        sources = "  \n".join(sources_list)
                        source_msg = "  \n  \n***知识库引用***  \n" + sources
                        
                        for c in source_msg:
                            response_msg += c
                            response_container.write(response_msg)
                
                message = {"role": "assistant", "content": response_msg}
                st.session_state.messages.append(message) # Add response to message history
                
                # log nonnull converstaion to langsmith
                _ = '''if prompt and response_msg:
                    print(f'{prompt} -> {response_msg}')
                    requests.patch(
                        f"https://api.smith.langchain.com/runs/{run_id}",
                        json={
                            "name": st.session_state.session_id,
                            "inputs": {"text": prompt},
                            "outputs": {"my_output": response_msg},
                        },
                        headers={"x-api-key": langchain_api_key},
                    )
                ''' 
                # st.rerun()
                _ = '''
                with tracing_v2_enabled(os.environ["LANGCHAIN_PROJECT"]) as cb:
                    feedback_index = int(
                        (len(st.session_state.messages) - 1) / 2
                    )
                    st.session_state[f"run_{feedback_index}"] = run.id
                    run = cb.latest_run
                    streamlit_feedback(**feedback_kwargs, key=f"feedback_{feedback_index}")
                '''
            # clear starter prompts upon convo
            if len(st.session_state.messages)==2:
                st.rerun()
        
main()
