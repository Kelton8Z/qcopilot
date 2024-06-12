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


app_id = st.secrets.feishu_app_id
app_secret = st.secrets.feishu_app_secret
# node_id = os.environ['NODEID']
# FeishuWikiReader = download_loader("FeishuWikiReader")
# FeishuWikiReader.wiki_spaces_url_path = "/open-apis/wiki/v2/spaces/{}/nodes"
# loader = FeishuWikiReader(app_id, app_secret)

client = lark.Client.builder() \
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
    response: InternalAppAccessTokenResponse = client.auth.v3.app_access_token.internal(request)

    # 处理失败返回
    if not response.success():
        lark.logger.error(
            f"client.auth.v3.app_access_token.internal failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}")
        return

    # 处理业务结果
    lark.logger.info(lark.JSON.marshal(response.data, indent=4))

    return response.data["app_access_token"]

def getOAuthCode(app_id, redirect_url):
    url = f"https://open.feishu.cn/open-apis/authen/v1/authorize?app_id={app_id}&redirect_uri={redirect_url}"
    headers = {
        "Content-Type": "application/json"
    }
    response = requests.get(url, headers)
    if response.status_code == 200:
        return response.data["code"]
    else:
        print("Failed to get oauth code. Status code:", response.status_code)

def getTenantAccessToken(app_id, app_secret):
    request: InternalTenantAccessTokenRequest = InternalTenantAccessTokenRequest.builder() \
        .request_body(InternalTenantAccessTokenRequestBody.builder()
            .app_id(app_id)
            .app_secret(app_secret)
            .build()) \
        .build()

    # 发起请求
    response: InternalTenantAccessTokenResponse = client.auth.v3.tenant_access_token.internal(request)

    # 处理失败返回
    if not response.success():
        lark.logger.error(
            f"client.auth.v3.tenant_access_token.internal failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}")
        return

    # 处理业务结果
    lark.logger.info(lark.JSON.marshal(response, indent=4))
    return json.loads(response.raw.content)["tenant_access_token"]

def getUserAccessToken(oauth_code):
    # 构造请求对象
    request: CreateOidcAccessTokenRequest = CreateOidcAccessTokenRequest.builder() \
        .request_body(CreateOidcAccessTokenRequestBody.builder()
            .grant_type("authorization_code")
            .code(oauth_code)
            .build()) \
        .build()

    # 发起请求
    response: CreateOidcAccessTokenResponse = client.authen.v1.oidc_access_token.create(request)

    # 处理失败返回
    if not response.success():
        lark.logger.error(
            f"client.authen.v1.oidc_access_token.create failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}")
        return

    # 处理业务结果
    lark.logger.info(lark.JSON.marshal(response.data, indent=4))

    return response.data["access_token"]

# get oauth code to get user access token
redirect_url = "https://open.feishu.cn/api-explorer/cli_a6df1d71d5f2d00d"
# app_access_token = getAppAccessToken(app_id, app_secret)
# oauth_code = getOAuthCode(app_id, redirect_url)
# user_access_token = getUserAccessToken(oauth_code)

def readWiki(space_id, app_id, app_secret):
    tenant_access_token = getTenantAccessToken(app_id, app_secret)

    # list wiki docs
    request: ListSpaceNodeRequest = ListSpaceNodeRequest.builder() \
        .space_id(space_id) \
        .build()

    # 发起请求
    response: ListSpaceNodeResponse = client.wiki.v2.space_node.list(request)

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

    docIDs = [item.obj_token for item in response.data.items]
    # print(f'DOC IDs: {docIDs}')
    docTitles = [item.title for item in response.data.items]

    # read docs

    for doc_id, title in zip(docIDs, docTitles):

        # 构造请求对象
        request: GetDocumentRequest = GetDocumentRequest.builder() \
            .document_id(doc_id) \
            .build()

        # 发起请求
        option = lark.RequestOption.builder().tenant_access_token(tenant_access_token).build()
        # response: GetDocumentResponse = client.docx.v1.document.get(request, option)

        # # 处理失败返回
        # if not response.success():
        #     lark.logger.error(
        #         f"client.docx.v1.document.get failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}")
        #     return

        # # 处理业务结果
        # lark.logger.info(lark.JSON.marshal(response.data, indent=4))

        # title = response.data.document.title
        # doc_id = response.data.document.id

        request: RawContentDocumentRequest = RawContentDocumentRequest.builder() \
            .document_id(doc_id) \
            .lang(0) \
            .build()

        # 发起请求
        response: RawContentDocumentResponse = client.docx.v1.document.raw_content(request, option)
        if not response.success():
            lark.logger.error(
                f"client.docx.v1.document.get failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}")
        else:
            with open("./data/"+title, 'w') as f:
                f.write(response.data.content)

        request: ListDocumentBlockRequest = ListDocumentBlockRequest.builder() \
        .document_id(doc_id) \
        .page_size(500) \
        .document_revision_id(-1) \
        .build()

        # 发起请求
        response: ListDocumentBlockResponse = client.docx.v1.document_block.list(request)

        # 处理失败返回
        if not response.success():
            lark.logger.error(
                f"client.docx.v1.document_block.list failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}")
        else:
            # 处理业务结果
            lark.logger.info(lark.JSON.marshal(response.data, indent=4))
        
    

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