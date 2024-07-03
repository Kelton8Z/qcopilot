from datetime import datetime, timedelta
from langsmith import Client
import streamlit as st
import os 
from tqdm import tqdm

langchain_api_key = os.environ["LANGCHAIN_API_KEY"] = st.secrets.langsmith_key

langsmith_project_id = st.secrets.langsmith_project_id
langsmith_client = Client(api_key=langchain_api_key)

client = Client()

start_time = datetime.utcnow() - timedelta(days=10)

runs = list(
    client.list_runs(
        project_name="stage",
        run_type="chain",
        start_time=start_time,
    )
)

print(f'{len(runs)} runs')

runs = [run for run in runs if not run.error]
print(f'{len(runs)} with no error')

demo_prompts = ["应该如何衡量decode和prefill过程的性能?", "AI Infra行业发展的目标是什么?", "JSX有什么优势?", "怎么实现capcha/防ai滑块?", "官网首页展示有哪些前端方案?", "我们的前端开发面试考察些什么?", "介绍一下CHT830项目", "llama模型平均吞吐量(token/s)多少?"]
runs = [run for run in runs if (run.inputs and run.inputs not in demo_prompts)] #and (run.outputs and ("output" in run.outputs and run.outputs["output"]))
print(f'{len(runs)} with UGC IO')

import pandas as pd

df = pd.DataFrame(
    [
        {
            **runs[i].inputs,
            **(runs[i].outputs or {}),
        }
        for i in tqdm(range(len(runs)))
    ],
    index=[run.id for run in runs],
)

print(df.head(5))
csv_file_path = 'stage.csv'
df.to_csv(csv_file_path, index=False)