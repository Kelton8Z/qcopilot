import lark_oapi as lark
from lark_oapi.api.wiki.v2 import *
from lark_oapi.api.docx.v1 import *
from lark_oapi.api.drive.v1 import *
from lark_oapi.api.auth.v3 import *
from lark_oapi.api.sheets.v3 import *
import streamlit as st

app_id = st.secrets.feishu_app_id
app_secret = st.secrets.feishu_app_secret

larkClient = lark.Client.builder() \
        .enable_set_token(True) \
        .log_level(lark.LogLevel.DEBUG) \
        .app_id(app_id) \
        .app_secret(app_secret) \
        .build()
        
sheet_token = "I869s9aRQhixhdt8bq1cveIznCb"
'''
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
'''
    
import requests
tenant_access_token = "t-g10485e4MHIYN4J5BV5JGOFXYFRNT3L2YEK4FIVI" #getTenantAccessToken(app_id, app_secret)
sheet_id = "ZcvSKH"
url = f'https://open.feishu.cn/open-apis/sheets/v2/spreadsheets/{sheet_token}/values/{sheet_id}'
headers = {
    'Authorization': f'Bearer {tenant_access_token}'
}

response = requests.get(url, headers=headers)
if response.status_code==200:
    respJson = response.json()
    sheet_data = respJson["data"]["valueRange"]["values"]
    # df = pd.DataFrame(sheet_data)
    # df.to_excel(writer, sheet_name=sheet.title, index=False)
else:
    lark.logger.error(
    f"Getting sheet from {url} failed, code: {response.status_code}, msg: {response.text}")