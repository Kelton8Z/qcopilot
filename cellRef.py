import os 
import time
import json
import requests
import lark_oapi as lark
from lark_oapi.api.wiki.v2 import *
from lark_oapi.api.docx.v1 import *
from lark_oapi.api.drive.v1 import *
from lark_oapi.api.auth.v3 import *
from lark_oapi.api.sheets.v3 import *
import streamlit as st
from listAllWiki import *
from readFeishuWiki import getUrl, readUrl, getFeishuNode, getFeishuSheets, getFeishuDocIDfromUrl, getTenantAccessToken
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.core.readers.base import BaseReader
import pandas as pd
from llama_index.core import Document, Settings, get_response_synthesizer
from llama_index.vector_stores.chroma import ChromaVectorStore

import chromadb
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core.schema import TextNode

# class EmbeddingTextNode(TextNode):
#     def __init__(self, text, *args, **kwargs):
#         super().__init__(text, *args, **kwargs)
#         self.embedding_model = embed_model
#         self._embedding = None
    
class CustomExcelReader(BaseReader):
    def load_data(self, filename: str, extra_info: dict = None):
        df = pd.read_excel(filename, sheet_name=None)
        sentences = []
        url = extra_info["file_url"] # e.g. https://chitu-ai.feishu.cn/wiki/SphbwVnJ6iSGCKk80V2cMhBHnQd
        doc_id = getFeishuDocIDfromUrl(url)
        sheetIDs = []
        response = getFeishuNode(doc_id)
        if response:
            sheet_token = response.data.node.obj_token
            sheets = getFeishuSheets(sheet_token)
            if sheets:
                sheetIDs = [sheet.sheet_id for sheet in sheets] 
                sheetNames = [sheet.title for sheet in sheets] 
                assert(sheetNames==df.keys())
        
        # if {'Model', 'Input Length', 'Output Length', 'Batch Size', 'Latency'}.issubset(df.columns):
        for (sheetname, sheet), sheetID in zip(df.items(), sheetIDs):
            suffix = "?" + sheetID if sheetID else ""
            sheet_url = url + suffix
            cols = sheet.columns
            for row_index, row in sheet.iterrows():
                sentence = f"For {sheetname}, "
                for col_index, (cell, col) in enumerate(zip(row, cols)):
                    if pd.notna(cell):
                        # Calculate cell location (e.g., A2, B2, etc.)
                        col_letter = chr(65 + col_index)  # Convert column index to letter
                        cell_location = f"{col_letter}{row_index + 2}"
                        sentence += f'{col} is {cell} (Location: [{sheetname}]({sheet_url}) {cell_location}), '

                sentences.append(sentence)

        return [Document(text=sentence, metadata=extra_info) for sentence in sentences]
    
def get_embedding(self):
    """
    Generate and return the embedding for the text content of this node.
    If the embedding is already computed, return it directly.
    """
    if self._embedding is None:
        self._embedding = self.embedding_model.encode(self.text)
    return self._embedding

api_version = "2024-05-01-preview"
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

Settings.embed_model = embed_model

azure_openai = AzureOpenAI( 
    engine=azure_chat_deployment,
    temperature=0,
    api_key=azure_api_key,  
    api_version=api_version,
    azure_endpoint = azure_endpoint,
)

fileToTitleAndUrl = {}
app_id = st.secrets.feishu_app_id
app_secret = st.secrets.feishu_app_secret
larkClient = lark.Client.builder() \
        .enable_set_token(True) \
        .log_level(lark.LogLevel.DEBUG) \
        .app_id(app_id) \
        .app_secret(app_secret) \
        .build()
        
llm_map = {"gpt4o": AzureOpenAI(model="gpt-4o", 
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
            ),
}
llm = llm_map["gpt4o"]
response_synthesizer = get_response_synthesizer(llm=llm, response_mode="compact", structured_answer_filtering=True)
# compact/refine with structured_answer_filtering=True to filter out any input nodes that are not relevant to the question being asked.

def main():
    directory = "a"
    url = "https://chitu-ai.feishu.cn/wiki/SphbwVnJ6iSGCKk80V2cMhBHnQd?from=from_copylink"

    # vector_store = azure_openai.beta.vector_stores.create(name="Model Performances")

    collection = "cell"
    chroma_db_path = "cell"

    db = chromadb.PersistentClient(path=chroma_db_path)
    chroma_collection = db.get_or_create_collection(collection)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model,
    )
    print(f'index from chroma {index}')
    reader = SimpleDirectoryReader(
        input_dir=directory,
        recursive=True,
        filename_as_id=True,
        file_extractor={".xlsx": CustomExcelReader()},
        file_metadata=lambda filename: {"file_name": filename, "file_url": url},
        raise_on_error=True
    )

    docs = reader.load_data()
    index = VectorStoreIndex([])
    if docs:
        try:
            index = VectorStoreIndex.from_documents(docs, embed_model=embed_model)
            # st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", llm=llm_map[st.session_state["llm"]], streaming=True)
            success = True
        except:
            success = False

    query_engine = index.as_query_engine(response_mode="no_text")
    query = "what is llama3 8b's throughput"
    resp = query_engine.query(query)
    # sys.stdout = original_stdout

    sources = resp.source_nodes
    
    from llama_index.core import PromptTemplate
    qa_prompt = PromptTemplate(
        """\
    Context information is below.
    ---------------------
    {context_str}
    ---------------------
    Given the context information and not prior knowledge, answer the query.
    Query: {query_str}
    Answer: \
    """
    )
    
    refine_prompt = PromptTemplate(
        """\
    The original query is as follows: {query_str}
    We have provided an existing answer: {existing_answer}
    We have the opportunity to refine the existing answer \
    (only if needed) with some more context below.
    ------------
    {context_str}
    ------------
    Given the new context, refine the original answer to better answer the query. \
    If the context isn't useful, return the original answer.
    If data retrieval or analysis is involved, show the source table and location of data within, i.e. keep the location data as in Throughput（requests/s) is 24.15 (Location: D13).
    Refined Answer: \
    """
    )
    from customResponseSynthesis import generate_response_cr
    cr_resp, fmt_prompts = generate_response_cr(sources, query, qa_prompt, refine_prompt, llm)
    print(f'answer {cr_resp}')
    print(f'intermediates {fmt_prompts}')
    
main()