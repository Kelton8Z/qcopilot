import os 
import requests
import json
# from llama_index.readers.download import download_loader
from llama_index.core import download_loader
import lark_oapi as lark
from lark_oapi.api.wiki.v2 import *
from lark_oapi.api.docx.v1 import *

app_id = os.environ['APPID']
app_secret = os.environ['APPSECRET']
node_id = os.environ['NODEID']
doc_id = os.environ['DOCID'] 
FeishuWikiReader = download_loader("FeishuWikiReader")
FeishuWikiReader.wiki_spaces_url_path = "/open-apis/wiki/v2/spaces/{}/nodes"
loader = FeishuWikiReader(app_id, app_secret)

user_access_token = os.environ['USERACCESSTOKEN']
def readWiki(space_id):
    client = lark.Client.builder() \
        .enable_set_token(True) \
        .log_level(lark.LogLevel.DEBUG) \
        .build()

    # 构造请求对象
    request: ListSpaceNodeRequest = ListSpaceNodeRequest.builder() \
        .space_id(space_id) \
        .build()

    # 发起请求
    option = lark.RequestOption.builder().user_access_token(user_access_token).build()
    response: ListSpaceNodeResponse = client.wiki.v2.space_node.list(request, option)

    # 处理失败返回
    if not response.success():
        lark.logger.error(
            f"client.wiki.v2.space_node.list failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}")
        return

    # 处理业务结果
    lark.logger.info(lark.JSON.marshal(response.data, indent=4))


    # 构造请求对象
    # request: SearchWikiRequest = SearchWikiRequest.builder() \
    #     .query("Case 1") \
    #     .space_id(space_id) \
    #     .node_id(node_id) \
    #     .build()
    # response: SearchWikiResponse = client.wiki.v2.space_node.search(request, option)

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
    option = lark.RequestOption.builder().user_access_token(user_access_token).build()
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
    
    

    # # 处理失败返回
    # if not response.success():
    #     lark.logger.error(
    #         f"client.wiki.v2.space_node.list failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}")
    #     return

    # # 处理业务结果
    # lark.logger.info(lark.JSON.marshal(response.data, indent=4))

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

# readWiki(space_id)
# query = "Case 1"
# searchWiki(space_id, node_id, query, user_access_token)