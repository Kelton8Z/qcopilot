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
from multiprocessing import Process, Queue, set_start_method
from functools import partial
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.llms.anthropic import Anthropic
from llama_index.core import Settings, SimpleDirectoryReader, get_response_synthesizer 
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core.postprocessor import SimilarityPostprocessor

from readFeishuWiki import readWiki, readUrl, read_documents, CustomExcelReader, TsvReader, getTenantAccessToken
from S3ops import put_object, upload_file, create_bucket, create_presigned_url, delete_all_objects, delete_bucket, bucket_exists

from streamlit_feedback import streamlit_feedback
from langsmith.run_helpers import get_current_run_tree
from langchain_core.tracers.context import tracing_v2_enabled
from langsmith import Client, traceable

Settings.chunk_size = 256

title = "AI assistant, powered by Qingcheng knowledge"
st.set_page_config(page_title=title, page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)

llama3_api_base = "http://localhost:25121/v1"
openai_api_base = "http://vasi.chitu.ai/v1"

os.environ["AWS_ACCESS_KEY_ID"] = st.secrets.aws_access_key
os.environ["AWS_SECRET_ACCESS_KEY"] = st.secrets.aws_secret_key

os.environ["LANGCHAIN_PROJECT"] = "august"
os.environ["LANGCHAIN_TRACING_V2"] = "true" 
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
langchain_api_key = os.environ["LANGCHAIN_API_KEY"] = st.secrets.langsmith_key
langsmith_client = Client(api_key=langchain_api_key)

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
if 'uploaded_url' not in st.session_state:
    st.session_state.uploaded_url = ""
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

if 'session_id' not in st.session_state or not st.session_state.session_id:
    st.session_state['session_id'] = str(uuid.uuid4())

prompt = "You are an expert ai infra analyst at 清程极智. Use your knowledge base to answer questions about ai model/hardware performance. After each line, show URL of your source and provide a confidence score"

os.environ["ANTHROPIC_API_KEY"] = st.secrets.claude_key
os.environ["JINAAI_API_KEY"] = st.secrets.jinaai_key

st.title(title)

def _submit_feedback(user_response, emoji=None, run_id=None):
    feedback = user_response['score']
    feedback_text = user_response['text']
    st.toast(f"Feedback submitted: {feedback}", icon=emoji)
    messages = st.session_state.messages
    if len(messages)>1:
        langsmith_client.create_feedback(
            run_id,
            key="user-score",
            score=0.0 if feedback=="👎" else 1.0,
            comment=feedback_text,
        )
    return user_response

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the docs – hang tight! This should take 1-2 minutes."):
        app_id = st.secrets.feishu_app_id
        app_secret = st.secrets.feishu_app_secret
        index, fileToTitleAndUrl = asyncio.run(readWiki(space_id, app_id, app_secret, embed_model))
        
        return index, fileToTitleAndUrl 
 
llm_map = {"Claude3.5": Anthropic(model="claude-3-5-sonnet-20240620", system_prompt=prompt), 
           "gpt4o": AzureOpenAI(model="gpt-4o", 
                system_prompt=prompt,
                engine=azure_chat_deployment,
                temperature=0,
                api_key=azure_api_key,  
                api_version=api_version,
                azure_endpoint = azure_endpoint,
            ),
           "gpt3.5":  AzureOpenAI(model="gpt-3.5-turbo", 
                engine=azure_chat_deployment,
                api_key=azure_api_key,  
                api_version=api_version,
                azure_endpoint = azure_endpoint,
                system_prompt=prompt
            ),
           "Llama3_8B": OpenAI(api_base=llama3_api_base, api_key="aba", system_prompt=prompt),
           "ollama": Ollama(model="llama2", request_timeout=60.0)
}
llm = "gpt4o"
response_synthesizer = get_response_synthesizer(llm=llm_map[llm], response_mode="compact")

def toggle_llm():
    llm = st.sidebar.selectbox(
        "模型切换",
        ("gpt4o", "Claude3.5", "Llama3_8B")
    )
    response_synthesizer = get_response_synthesizer(llm=llm_map[llm], response_mode="compact")
    if llm=="Llama3_8B": 
        os.environ["OPENAI_API_BASE"] = llama3_api_base
    
    if llm!=st.session_state["llm"]:
        st.session_state["llm"] = llm
        Settings.llm = llm_map[llm]
        st.toast(f'Switched to {llm}')


def toggle_rag_use():
    
    from llama_index.core import VectorStoreIndex, Document
        
    use_rag = st.sidebar.selectbox(
        "是否用飞书知识库",
        ("否", "是")
    )
    use_rag = True if use_rag=="是" else False
   
    if use_rag!= st.session_state.use_rag:
        st.session_state.chat_engine = SimpleChatEngine.from_defaults(llm=llm_map[st.session_state["llm"]], system_prompt=prompt)
        if use_rag:
            index, fileToTitleAndUrl = load_data()                
            if index:
                st.session_state.fileToTitleAndUrl = fileToTitleAndUrl 
                st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", llm=llm_map[st.session_state["llm"]], response_synthesizer=response_synthesizer, streaming=True)
            else:
                st.toast("调用飞书知识库失败")
                use_rag = False
        
        st.session_state.use_rag = use_rag
        st.rerun()
    else:
        if "chat_engine" not in st.session_state.keys():
            st.session_state.chat_engine = SimpleChatEngine.from_defaults(llm=llm_map[st.session_state["llm"]], system_prompt=prompt)
            if use_rag:
                index, fileToTitleAndUrl = load_data()                
                if index:
                    st.session_state.fileToTitleAndUrl = fileToTitleAndUrl 
                    st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", llm=llm_map[st.session_state["llm"]], streaming=True)

    if not use_rag:
        uploaded_files = st.sidebar.file_uploader(label="上传临时文件", accept_multiple_files=True)
        uploaded_url = st.sidebar.text_input("飞书文件链接")
        docs = []
        directory = st.session_state.session_id
        if st.session_state.uploaded_url!=uploaded_url:
            st.session_state.uploaded_url = uploaded_url
            if uploaded_url:
                with st.status(label="读取链接中", expanded=False) as status:
                    tenant_access_token = getTenantAccessToken(app_id, app_secret)
                    directory = st.session_state.session_id
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    resp = readUrl(directory, uploaded_url, tenant_access_token)
                    reader = SimpleDirectoryReader(
                        input_dir=directory,
                        recursive=True,
                        filename_as_id=True,
                        file_extractor={".xlsx": CustomExcelReader(), ".tsv": TsvReader()},
                        file_metadata=lambda filename: {"file_name": filename},
                        raise_on_error=True
                    )
                    docs = reader.load_data()
                    if docs:
                        try:
                            index = VectorStoreIndex.from_documents(docs, embed_model=embed_model)
                            st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", llm=llm_map[st.session_state["llm"]], streaming=True)
                            success = True
                        except:
                            success = False

                    if success:
                        status.update(label="读取完成", state="complete", expanded=False)
                    else:
                        status.update(label="读取失败", state="error", expanded=False)

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
                            #s3_url = create_presigned_url(bucket_name=directory, object_name=filename)
                            s3_url = ""
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
                        uploaded_docs = queue.get()
                        if uploaded_docs: 
                            docs.extend(uploaded_docs)

                        # Ensure the subprocess is terminated
                        if process.is_alive():
                            process.terminate()

                        if docs:
                            try:
                                index = VectorStoreIndex.from_documents(docs, embed_model=embed_model)
                                st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", llm=llm_map[st.session_state["llm"]], streaming=True)
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
        #st.session_state.chat_engine.reset()
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

map = {}
@traceable(run_type="llm", name=st.session_state.session_id)
def log_trace(prompt):
    run = get_current_run_tree()
    run_id = str(run.id)
    st.session_state.run_id = st.session_state["run_0"] = run_id
    return map[prompt], run_id

def call_llm(prompt):
    streaming_response = st.session_state.chat_engine.stream_chat(prompt)
    return streaming_response

def main():
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
    for message in st.session_state.messages: 
        with st.chat_message(message["role"]):
            st.write(message["content"])
    

    if st.session_state.messages:
        message = st.session_state.messages[-1]
        if message["role"]=="assistant":
            i = len(st.session_state.messages)-1
            feedback_key = f"feedback_{int(i/2)}"
            # This actually commits the feedback
            streamlit_feedback(
                **feedback_kwargs,
                key=feedback_key,
                on_submit=partial(
                    _submit_feedback, run_id=st.session_state[f"run_{int(i/2)}"]
                ),
            )

        # If last message is not from assistant, generate a new response
        if message["role"] != "assistant":
            with st.chat_message("assistant"):
                response_container = st.empty()  # Container to hold the response as it streams
                response_msg = ""
                
                if prompt:
                    streaming_response = call_llm(prompt)
                else:
                    st.rerun()
                
                for token in streaming_response.response_gen:
                    response_msg += token
                    response_container.write(response_msg)
            
                try:
                    map[prompt] = response_msg
                    _, run_id = log_trace(prompt)
                except:
                    st.rerun()
                
                if st.session_state.use_rag or st.session_state.uploaded_files:
                    processor = SimilarityPostprocessor(similarity_cutoff=0.25)
                    source_nodes = streaming_response.source_nodes
                    filtered_nodes = processor.postprocess_nodes(source_nodes)
                    if source_nodes:
                        sources = set()
                        for node in source_nodes:
                            try:
                                if "file_path" in node.metadata:
                                    file_path = node.metadata["file_path"]
                                else:
                                    file_path = node.metadata["file_name"]
                        
                                file_url = st.session_state.fileToTitleAndUrl[file_path]["url"]
                                if file_url and file_path in st.session_state.fileToTitleAndUrl:
                                    file_name = st.session_state.fileToTitleAndUrl[file_path]["title"]
                                    source = "[%s](%s)" % (file_name, file_url) #+ " 相似度" + format(node.score, ".2%") 
                                else:
                                    file_name = file_path.split("/")[-1]
                                    source = "%s" % file_name #+ " 相似度" + format(node.score, ".2%") 
                                
                                sources.add(source)
                            except Exception as e:
                                # no source wiki node
                                print(st.session_state.fileToTitleAndUrl)
                                print(e)
                                pass
                        
                        if sources: 
                            sources = "  \n".join(sources)
                            source_msg = "  \n  \n***知识库引用***  \n" + sources
                            
                            for c in source_msg:
                                response_msg += c
                                response_container.write(response_msg)
                    
                            #st.markdown(list(sources))
                
                message = {"role": "assistant", "content": response_msg}
                st.session_state.messages.append(message) # Add response to message history
                
                # log nonnull converstaion to langsmith
                if prompt and response_msg:
                    requests.patch(
                        f"https://api.smith.langchain.com/runs/{run_id}",
                        json={
                            "name": st.session_state.session_id,
                            "inputs": {"text": prompt},
                            "outputs": {"my_output": response_msg},
                        },
                        headers={"x-api-key": langchain_api_key},
                    )
                # st.rerun()
                with tracing_v2_enabled(os.environ["LANGCHAIN_PROJECT"]) as cb:
                    feedback_index = int(
                        (len(st.session_state.messages) - 1) / 2
                    )
                    st.session_state[f"run_{feedback_index}"] = run_id
                    run = cb.latest_run
                    streamlit_feedback(**feedback_kwargs, key=f"feedback_{feedback_index}")
            
            # clear starter prompts upon convo
            if len(st.session_state.messages)==2:
                st.rerun()
        
main()
