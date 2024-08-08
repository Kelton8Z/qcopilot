import lark_oapi as lark
from lark_oapi.api.sheets.v3 import *
from lark_oapi.api.drive.v1 import *


# SDK 使用说明: https://github.com/larksuite/oapi-sdk-python#readme
# 复制该 Demo 后, 需要将 "YOUR_APP_ID", "YOUR_APP_SECRET" 替换为自己应用的 APP_ID, APP_SECRET.
def readFeishu():
    # 创建client
    client = lark.Client.builder() \
        .app_id("cli_a6df1d71d5f2d00d") \
        .app_secret("hXXZeFBeK2wN1eIviHrSHhRTqBZV1m7T") \
        .log_level(lark.LogLevel.DEBUG) \
        .build()


    # sheetToken = ""
    # # 构造请求对象
    request: QuerySpreadsheetSheetRequest = QuerySpreadsheetSheetRequest.builder() \
        .spreadsheet_token(sheetToken) \
        .build()

    # 发起请求
    response: QuerySpreadsheetSheetResponse = client.sheets.v3.spreadsheet_sheet.query(request)

    # 处理失败返回
    if not response.success():
        lark.logger.error(
            f"client.sheets.v3.spreadsheet_sheet.query failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}")
        return

    # 处理业务结果
    lark.logger.info(lark.JSON.marshal(response.data, indent=4))
    
from llama_index.core.readers.base import BaseReader
import pandas as pd

class ExcelReader(BaseReader):
    def load_data(self, file_path: str, extra_info: dict = None):
        data = pd.read_excel(file_path).to_string()
        return [Document(text=data, metadata=extra_info)]

def vectorize():
    from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

    
    reader = SimpleDirectoryReader(input_dir="sheets", recursive=True, file_extractor={".xlsx": ExcelReader()})
    docs = reader.load_data()

if __name__ == "__main__":
    # readFeishu()
    vectorize()