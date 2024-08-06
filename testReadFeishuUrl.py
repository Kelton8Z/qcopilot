from readFeishuWiki import readUrl, getTenantAccessToken, ExcelReader, TsvReader
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
import streamlit as st

app_id = st.secrets.feishu_app_id
app_secret = st.secrets.feishu_app_secret

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
llm = AzureOpenAI(model="gpt-3.5-turbo", 
    engine=azure_chat_deployment,
    api_key=azure_api_key,  
    api_version=api_version,
    azure_endpoint = azure_endpoint,
)

directory = "a"
url = "https://chitu-ai.feishu.cn/wiki/DHiNwzMbXifdQukWhFFcdkCXnAU?from=from_copylink"
tenant_access_token = getTenantAccessToken(app_id, app_secret)
m = readUrl(directory, url, tenant_access_token)

reader = SimpleDirectoryReader(
            input_dir=directory,
            recursive=True,
            filename_as_id=True,
            file_extractor={".xlsx": ExcelReader(), ".tsv": TsvReader()},
            file_metadata=lambda filename: {"file_name": filename},
            raise_on_error=True
)
docs = reader.load_data()
print(docs)
index = VectorStoreIndex.from_documents(docs, embed_model=embed_model)
chat_engine = index.as_chat_engine(chat_mode="condense_question", llm=llm, streaming=True)
print(chat_engine.chat("张卓然七月工作了多久"))
