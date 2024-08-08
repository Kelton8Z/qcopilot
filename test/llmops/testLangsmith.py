# import os
# import openai
import streamlit as st
# from langsmith.wrappers import wrap_openai
# from langsmith import traceable

# os.environ["OPENAI_API_KEY"] = st.secrets.openai_key
# os.environ["LANGCHAIN_PROJECT"] = "default"
# os.environ["LANGCHAIN_TRACING_V2"] = "true" 
# langchain_api_key = os.environ["LANGCHAIN_API_KEY"] = st.secrets.langsmith_key

# # Auto-trace LLM calls in-context
# client = wrap_openai(openai.Client())

# @traceable # Auto-trace this function
# def pipeline(user_input: str):
#     result = client.chat.completions.create(
#         messages=[{"role": "user", "content": user_input}],
#         model="gpt-3.5-turbo"
#     )
#     return result.choices[0].message.content

# pipeline("Hello, world!")

import datetime
import uuid

import requests

langchain_api_key = st.secrets.langsmith_key
run_id = str(uuid.uuid4())
import time
while True:
    resp = requests.post(
        "https://api.smith.langchain.com/runs",
        json={
            "id": run_id,
            "name": "MyFirstRun",
            "run_type": "chain",
            "start_time": datetime.datetime.utcnow().isoformat(),
            "inputs": {"text": "Foo"},
        },
        headers={"x-api-key": langchain_api_key},
    )
    if resp.status_code == 200:
        break
    elif resp.status_code == 202:
        print("Processing... retrying in 5 seconds.")
        time.sleep(5)  # Wait before retrying
    else:
        print(resp)

while True:
    patch_resp = requests.patch(
        f"https://api.smith.langchain.com/runs/{run_id}",
        json={
            "outputs": {"my_output": "bs"},
        },
        headers={"x-api-key": langchain_api_key},
    )
    if patch_resp.status_code == 200:
        break
    elif patch_resp.status_code == 202:
        print("Processing... retrying in 5 seconds.")
        time.sleep(5)  # Wait before retrying
    else:
        print(patch_resp)

print(patch_resp)