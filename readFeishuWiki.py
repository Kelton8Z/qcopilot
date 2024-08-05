import os 
import requests
import json
import lark_oapi as lark
from lark_oapi.api.wiki.v2 import *
from lark_oapi.api.docx.v1 import *
from lark_oapi.api.drive.v1 import *
from lark_oapi.api.auth.v3 import *
from lark_oapi.api.sheets.v3 import *
import streamlit as st
from listAllWiki import *

from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.core.readers.base import BaseReader
import pandas as pd
from llama_index.core import Document

import chromadb

collection = "qcWiki"
chroma_db_path = "chroma_db"
fileToTitleAndUrl = {}
required_exts = [".py", ".md", ".txt", ".docx", ".json", ".tsv", ".csv", ".ppt", ".pptx", ".xlsx"]

class CustomExcelReader(BaseReader):
    def load_data(self, filename: str, extra_info: dict = None):
        df = pd.read_excel(filename, sheet_name=None)
        sentences = []
        # if {'Model', 'Input Length', 'Output Length', 'Batch Size', 'Latency'}.issubset(df.columns):
        for sheetname, sheet in df.items():
            cols = sheet.columns
            for row in sheet.itertuples(index=False, name=None):
                sentence = f"For {sheetname}, "
                for cell, col in zip(row, cols):
                    if pd.notna(cell):
                        sentence += f'{col} is {cell}, '

                sentences.append(sentence)

        return [Document(text=sentence, metadata=extra_info) for sentence in sentences]

class TsvReader(BaseReader):
    def load_data(self, file_path: str, extra_info: dict = None):
        data = pd.read_csv(file_path, sep='\t').to_string()
        return [Document(text=data, metadata=extra_info)]

class ExcelReader(BaseReader):
    def load_data(self, file_path: str, extra_info: dict = None):
        data = pd.read_excel(file_path).to_string()
        return [Document(text=data, metadata=extra_info)]
    
def getUrl(client, doc_id, doc_type):
    request: BatchQueryMetaRequest = BatchQueryMetaRequest.builder() \
        .request_body(MetaRequest.builder()
            .request_docs([RequestDoc.builder()
                .doc_token(doc_id)
                .doc_type(doc_type)
                .build()
                ])
            .with_url(True)
            .build()) \
        .build()

    # 发起请求
    response: BatchQueryMetaResponse = client.drive.v1.meta.batch_query(request)

    # 处理失败返回
    if not response.success():
        lark.logger.error(
            f"client.drive.v1.meta.batch_query failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}")
        return None

    # 处理业务结果
    lark.logger.info(lark.JSON.marshal(response.data, indent=4))
    if len(response.data.metas)!=1:
        print(f'{doc_type}" -> {response.data.metas}')
    return response.data.metas[0].url

app_id = st.secrets.feishu_app_id
app_secret = st.secrets.feishu_app_secret

larkClient = lark.Client.builder() \
        .enable_set_token(True) \
        .log_level(lark.LogLevel.DEBUG) \
        .app_id(app_id) \
        .app_secret(app_secret) \
        .build()

def getAppAccessToken(app_id, app_secret):
    # 构造请求对象
    request: InternalAppAccessTokenRequest = InternalAppAccessTokenRequest.builder() \
        .request_body(InternalAppAccessTokenRequestBody.builder()
            .app_id(app_id)
            .app_secret(app_secret)
            .build()) \
        .build()

    # 发起请求
    response: InternalAppAccessTokenResponse = larkClient.auth.v3.app_access_token.internal(request)

    # 处理失败返回
    if not response.success():
        lark.logger.error(
            f"client.auth.v3.app_access_token.internal failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}")
        return

    # 处理业务结果
    lark.logger.info(lark.JSON.marshal(response.data, indent=4))

    return response.data["app_access_token"]


def getTenantAccessToken(app_id, app_secret):
    request: InternalTenantAccessTokenRequest = InternalTenantAccessTokenRequest.builder() \
        .request_body(InternalTenantAccessTokenRequestBody.builder()
            .app_id(app_id)
            .app_secret(app_secret)
            .build()) \
        .build()

    # 发起请求
    response: InternalTenantAccessTokenResponse = larkClient.auth.v3.tenant_access_token.internal(request)

    # 处理失败返回
    if not response.success():
        lark.logger.error(
            f"client.auth.v3.tenant_access_token.internal failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}")
        return

    # 处理业务结果
    lark.logger.info(lark.JSON.marshal(response, indent=4))
    return json.loads(response.raw.content)["tenant_access_token"]


def read_documents(directory, s3fs, queue, max_retries=5, retry_delay=2):
    for attempt in range(max_retries):
        try:
            reader = SimpleDirectoryReader(
                input_dir=directory,
                fs=s3fs,
                recursive=True,
                filename_as_id=True,
                required_exts = required_exts,
                file_extractor={".xlsx": CustomExcelReader(), ".tsv": TsvReader()},
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


async def readWiki(space_id, app_id, app_secret, embed_model):
    tenant_access_token = getTenantAccessToken(app_id, app_secret)
    nodes = await get_all_wiki_nodes(space_id, tenant_access_token)
    directory = "./data"
    if not os.path.exists(directory):
        os.makedirs(directory)

    for node in nodes:
        doc_id = node["obj_token"]
        title = node["title"]
        doc_type = node["obj_type"]

        # 发起请求
        option = lark.RequestOption.builder().tenant_access_token(tenant_access_token).build()

        if doc_type=="sheet":
            sheet_token = doc_id
            request: QuerySpreadsheetSheetRequest = QuerySpreadsheetSheetRequest.builder() \
        .spreadsheet_token(sheet_token) \
        .build()

            # 发起请求
            response: QuerySpreadsheetSheetResponse = larkClient.sheets.v3.spreadsheet_sheet.query(request)

            # 处理失败返回
            if not response.success():
                lark.logger.error(
                    f"client.sheets.v3.spreadsheet_sheet.query failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}")
            else:
                # 处理业务结果
                lark.logger.info(lark.JSON.marshal(response.data, indent=4))
                sheets = response.data.sheets

                with pd.ExcelWriter("./data/"+title, engine='xlsxwriter') as writer:
                    for sheet in sheets:
                        url = f'https://open.feishu.cn/open-apis/sheets/v2/spreadsheets/{sheet_token}/values/{sheet.sheet_id}'
                        headers = {
                            'Authorization': f'Bearer {tenant_access_token}'
                        }

                        response = requests.get(url, headers=headers)
                        if response.status_code==200:
                            respJson = response.json()
                            sheet_data = respJson["data"]["valueRange"]["values"]
                            df = pd.DataFrame(sheet_data)
                            df.to_excel(writer, sheet_name=sheet.title, index=False)
                        else:
                            lark.logger.error(
                            f"Getting sheet from {url} failed, code: {response.status_code}, msg: {response.text}")
            
        elif doc_type=="docx":
            request: RawContentDocumentRequest = RawContentDocumentRequest.builder() \
            .document_id(doc_id) \
            .lang(0) \
            .build()

            # 发起请求
            response: RawContentDocumentResponse = larkClient.docx.v1.document.raw_content(request, option)
            if not response.success():
                lark.logger.error(
                    f"client.docx.v1.document.get failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}, doc_id: {doc_id}")
            else:
                path = "./data/"+title
                if not os.path.exists(path):
                    try:
                        with open(path, 'w') as f:
                            f.write(response.data.content)
                    except:
                        pass

            request: ListDocumentBlockRequest = ListDocumentBlockRequest.builder() \
            .document_id(doc_id) \
            .page_size(500) \
            .document_revision_id(-1) \
            .build()

            # 发起请求
            listBlockResponse: ListDocumentBlockResponse = larkClient.docx.v1.document_block.list(request)

            # 处理失败返回
            if not listBlockResponse.success():
                lark.logger.error(
                    f"client.docx.v1.document_block.list failed, code: {listBlockResponse.code}, msg: {listBlockResponse.msg}, log_id: {listBlockResponse.get_log_id()}")
            else:
                # 处理业务结果
                lark.logger.info(lark.JSON.marshal(listBlockResponse.data, indent=4))
                
            try:
                fileToTitleAndUrl[os.path.abspath(path)] = {"title": title, "url": getUrl(larkClient, doc_id, doc_type)}
            except:
                print(f'failed {path}')
    
    # automatically sets the metadata of each document according to filename_fn
    reader = SimpleDirectoryReader(
                input_dir=directory, 
                recursive=True, 
                filename_as_id=True,
                required_exts = required_exts,
                file_extractor={".xlsx": CustomExcelReader()}, 
                file_metadata=lambda filename: {"file_name": filename}
            )
    docs = reader.load_data()
    
    import hashlib
    hash_file_path = "./documents_hash.txt"

    def calculate_hash(documents):
        """Calculate the SHA-256 hash of the documents."""
        doc_str = json.dumps([doc.text for doc in documents], sort_keys=True)
        return hashlib.sha256(doc_str.encode('utf-8')).hexdigest()

    def load_hash():
        """Load the previously saved hash from file."""
        if os.path.exists(hash_file_path):
            with open(hash_file_path, 'r') as f:
                return f.read()
        return None

    def save_hash(new_hash):
        """Save the new hash to file."""
        with open(hash_file_path, 'w') as f:
            f.write(new_hash)

    current_hash = calculate_hash(docs)

    # Load the previous hash
    previous_hash = load_hash()

    # Check if the documents have been updated
    if current_hash == previous_hash:
        print("Documents have not been updated. Loading from disk.")
        db = chromadb.PersistentClient(path=chroma_db_path)
        chroma_collection = db.get_or_create_collection(collection)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Load the existing index from the vector store
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            embed_model=embed_model,
        )
    else:
        print("Documents have been updated. Creating new vector store and index.")
        # Save the new hash
        save_hash(current_hash)

        # Create a new vector store and index
        db = chromadb.PersistentClient(path=chroma_db_path)
        chroma_collection = db.get_or_create_collection(collection)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Create a new index from documents
        index = VectorStoreIndex.from_documents(
            docs, storage_context=storage_context, embed_model=embed_model
        )
        
    return index, fileToTitleAndUrl


def searchWiki(space_id, node_id, query, user_access_token):

    # Define the URL and the headers
    url = "https://open.feishu.cn/open-apis/wiki/v1/nodes/search"
    headers = {
        "Content-Type": "application/json",
        "Authorization": user_access_token
    }

    # Define the request body parameters
    data = {
        "space_id": space_id,
        "node_id": node_id,
        "query": query
    }

    # Make the POST request
    response = requests.post(url, headers=headers, data=json.dumps(data))

    # Check the response
    if response.status_code == 200:
        result = response.json()
        print("Search Results:", result)
    else:
        print("Failed to search nodes. Status code:", response.status_code)
