import os 

# from llama_index.readers.download import download_loader
from llama_index.core import download_loader

app_id = os.environ['APPID']
app_secret = os.environ['APPSECRET']
space_id = os.environ['SPACEID']
FeishuWikiReader = download_loader("FeishuWikiReader")
FeishuWikiReader.wiki_spaces_url_path = "/open-apis/wiki/v2/spaces/{}/nodes"
loader = FeishuWikiReader(app_id, app_secret)
documents = loader.load_data(space_id=space_id)
print(documents)

import lark_oapi as lark
from lark_oapi.api.wiki.v2 import *

USERACCESSTOKEN = "u-faUN.2KI1fUVpKVJPrnFej502uKkh473qwG0lgcw2KIv"
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
    option = lark.RequestOption.builder().user_access_token(USERACCESSTOKEN).build()
    response: ListSpaceNodeResponse = client.wiki.v2.space_node.list(request, option)

    # 处理失败返回
    if not response.success():
        lark.logger.error(
            f"client.wiki.v2.space_node.list failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}")
        return

    # 处理业务结果
    lark.logger.info(lark.JSON.marshal(response.data, indent=4))

readWiki(space_id)