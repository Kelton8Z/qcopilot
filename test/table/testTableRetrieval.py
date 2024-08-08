import asyncio 
import pandas as pd
import openai
from llama_index.llms.openai import OpenAI
import anthropic
from llama_index.llms.anthropic import Anthropic
import os
import streamlit as st
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

api_version="2024-02-01"
azure_api_key = st.secrets.azure_api_key
azure_endpoint = st.secrets.azure_endpoint
azure_chat_deployment = st.secrets.azure_chat_deployment
azure_embedding_deployment = st.secrets.azure_embedding_deployment

llama3_api_base = "http://localhost:25121/v1"
os.environ["OPENAI_API_BASE"] = ""
os.environ["OPENAI_API_KEY"] = st.secrets.openai_key
openai_client = openai.OpenAI()
openai_assistant = openai_client.beta.assistants.create(
  name="AI Infra Analyst Assistant",
  instructions="You are an expert ai infra analyst. Use your knowledge base to answer questions about ai model/hardware performance.",
  model="gpt-4o",
  tools=[{"type": "code_interpreter"},{"type": "file_search"}]
)

os.environ["ANTHROPIC_API_KEY"] = st.secrets.claude_key
anthropic_client = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key=st.secrets.claude_key,
)

excel_file_path = '/Users/keltonzhang/Code/qc/a/模型性能对比.xlsx'
'''
create_file_response = openai_client.files.create(
    file=open(excel_file_path, 'rb'),
    purpose='assistants'
)
if create_file_response:
    file_id = create_file_response['id']
    print(f'openai file {file_id} created')
    content = openai_client.files.content(file_id)
'''


def test_function_performance(use_rag, model_type, model, input_len, output_len, throughput_requests, throughput_tokens, avg_latency_batch1, avg_latency_batch8, avg_latency_batch64, md_table="", message_file=None):
    ch_question = f'{model}模型输入长度为{input_len}输出长度为{output_len}时吞吐量(request/s)，吞吐量(tokens/s)，batch是1、8、64的平均延迟分别是多少? 只返回最终答案数字[Throughput（requests/s), Throughput（tokens/s), Avg latency（Batch=1）, Avg latency（Batch=8）, Avg latency（Batch=64）]'
    en_question = f'With input leng {input_len}, output length {output_len}, what are the Throughput（requests/s), Throughput（tokens/s), Avg latency（Batch=1）, Avg latency（Batch=8）, Avg latency（Batch=64）for {model}? Only return the numbers'
    
    '''
    questions = [
        f'{model}模型输入长度为{input_len}输出长度为{output_len}时吞吐量(request/s)是多少?',
        f'{model}模型输入长度为{input_len}输出长度为{output_len}时吞吐量(tokens/s)是多少?',
        f'{model}模型输入长度为{input_len}输出长度为{output_len}时batch是1的平均延迟是多少?',
        f'{model}模型输入长度为{input_len}输出长度为{output_len}时batch是8的平均延迟是多少?',
        f'{model}模型输入长度为{input_len}输出长度为{output_len}时batch是64的平均延迟是多少?'
    ]
    
    groundtruths = [str(throughput_requests), str(throughput_tokens), str(avg_latency_batch1), str(avg_latency_batch8), str(avg_latency_batch64)]
    cur_hit = 0
    for q, gt in zip(questions, groundtruths):
    '''
    msg = ""
    if model_type=="gpt4o":
        if use_rag=="yes":
            response = st.session_state.chat_engine.chat(en_question)
            msg = response.response
            print(msg)
        else:
            # Create a thread and attach the file to the message
            thread = openai_client.beta.threads.create(
              messages=[
                {
                  "role": "user",
                  "content": en_question,
                  "attachments": [
                    {
                      "file_id": message_file.id,
                      "tools": [{"type": "code_interpreter"}]
                    }
                  ]
                }
              ]
            )


            run = openai_client.beta.threads.runs.create_and_poll(
                thread_id=thread.id, assistant_id=openai_assistant.id
            )

            messages = list(openai_client.beta.threads.messages.list(thread_id=thread.id, run_id=run.id))

            message_content = messages[0].content[0].text
            msg = message_content.value
    elif model_type=="Claude3.5":
        if use_rag=="yes":
            while True:
                try:
                    response = st.session_state.chat_engine.chat(en_question)
                    msg = response.response
                    print(msg)
                    break
                except Exception as e:
                    print(e)
                    import time
                    time.sleep(60)
        else:
            query = md_table+"\n"+en_question
            message = anthropic_client.messages.create(
                model=claude3pt5, #"claude-3-sonnet-20240229",
                max_tokens=10,
                messages=[
                    {"role": "user", "content": query}
                ]
            )
            msg = message.content[0].text
            
    # cur_hit += int(gt in msg)
        
    cur_hit = sum([str(throughput_requests) in msg, str(throughput_tokens) in msg, str(avg_latency_batch1) in msg, str(avg_latency_batch8) in msg, str(avg_latency_batch64) in msg])      
    return cur_hit, 5-cur_hit
  
claude3pt5 = "claude-3-5-sonnet-20240620"
prompt = "You are an expert ai infra analyst at 清程极智. Use your knowledge base to answer questions about ai model/hardware performance. Show URLs of your sources whenever possible"
llm_map = {"Claude3.5": Anthropic(model=claude3pt5, system_prompt=prompt), 
           "gpt4o": AzureOpenAI(model="gpt-4o", 
               system_prompt=prompt,
                engine=azure_chat_deployment,
                api_key=azure_api_key,  
                api_version=api_version,
                azure_endpoint = azure_endpoint,
            ),
           "gpt3.5":  AzureOpenAI(model="gpt-3.5-turbo", 
                engine=azure_chat_deployment,
                api_key=azure_api_key,  
                api_version=api_version,
                azure_endpoint = azure_endpoint,
                system_prompt=prompt
            ),
           "Llama3_8B": OpenAI(base_url=llama3_api_base),
}

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the docs – hang tight! This should take 1-2 minutes."):
        space_id = st.secrets.feishu_space_id
        app_id = st.secrets.feishu_app_id
        app_secret = st.secrets.feishu_app_secret
        embed_model = AzureOpenAIEmbedding(
            model="text-embedding-ada-002",
            deployment_name=azure_embedding_deployment,
            api_key=azure_api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
        )
        import sys
        grandparent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   
        sys.path.append(grandparent_dir)     
        from readFeishuWiki import ExcelReader
        dir = "/Users/keltonzhang/Code/qc/data"
        reader = SimpleDirectoryReader(
                input_dir=dir, 
                recursive=True, 
                filename_as_id=True,
                file_extractor={".xlsx": ExcelReader()}, 
                # required_exts=[".xlsx"],
                file_metadata=lambda filename: {"file_name": filename}
        )
        docs = reader.load_data()
        index = VectorStoreIndex.from_documents(
            docs, embed_model=embed_model
        )
        # index, fileToTitleAndUrl = asyncio.run(readDocs(docs, embed_model)) #readWiki(embed_model))
        
        return index, {} #fileToTitleAndUrl 
  

def main(file_type, model_type, use_rag):
  # Ready the files for upload to LLM 
  file_path = excel_file_path
  df = pd.read_excel(excel_file_path)
  md_table = ""
  message_file = None
  if use_rag=="yes":
      index, fileToTitleAndUrl = load_data()         
      if index:
          st.session_state.fileToTitleAndUrl = fileToTitleAndUrl 
          st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", llm=llm_map[model_type], streaming=True)
  else:
      if file_type=="md":
          md_file_path = excel_file_path.split('.xlsx')[0]+'.md'
          file_path = md_file_path
          md_table = df.to_markdown(tablefmt="grid")
          with open(md_file_path, 'w') as file:
              file.write(md_table)
      elif file_type=="csv":
          csv_file_path = excel_file_path.split('.xlsx')[0]+'.csv'
          file_path = csv_file_path
          csv_table = df.to_csv()
          with open(csv_file_path, 'w') as file:
              file.write(csv_table)
    
  if model_type=="gpt4o":
      if use_rag=="no":
          vector_store = openai_client.beta.vector_stores.create(name="Model Performances")
          # Upload the user provided file to OpenAI
          message_file = openai_client.files.create(
            file=open(file_path, "rb"), purpose="assistants"
          )

          print(f'provided {message_file.id} to openai')
          global openai_assistant
          openai_assistant = openai_client.beta.assistants.update(
            assistant_id=openai_assistant.id,
            tool_resources={
                "file_search": {
                    "vector_store_ids": [vector_store.id]
                },
                "code_interpreter": {
                  "file_ids": [message_file.id]
                }
            },
          )
  # elif model_type=="claude3.5sonnet":

  
  # 假设Excel文件中有以下列：Model, Input, Output, Throughput_Requests, Throughput_Tokens, Avg_Latency_Batch1, Avg_Latency_Batch8, Avg_Latency_Batch64
  # 使用DataFrame中的数据测试函数
  hit = 0
  miss = 0
  from tqdm import tqdm
  for idx, row in tqdm(df.iterrows()):
      model = row['Model']
      input_len = row['Input']
      output_len = row['Output']
      throughput_requests = row['Throughput（requests/s)']
      throughput_tokens = row['Throughput（tokens/s)']
      avg_latency_batch1 = row['Avg latency（Batch=1）']
      avg_latency_batch8 = row['Avg latency（Batch=8）']
      avg_latency_batch64 = row['Avg latency（Batch=64）']
      cur_hit, cur_miss = test_function_performance(use_rag, model_type, model, input_len, output_len, throughput_requests, throughput_tokens, avg_latency_batch1, avg_latency_batch8, avg_latency_batch64, md_table, message_file)
      hit += cur_hit
      miss += cur_miss

      print(f'accuracy: {hit/(hit+miss)}')
  
import argparse
parser = argparse.ArgumentParser(description="Process some files with a specified model.")
parser.add_argument(
    '-f', '--file_type', 
    type=str, 
    required=True, 
    help='Specify the type of the file (csv, md, xlsx).'
)
parser.add_argument(
    '-m', '--model', 
    type=str, 
    required=True, 
    help='Specify the model to use (gpt4o, Claude3.5).'
)
parser.add_argument(
    '-rag', '--use_rag', 
    type=str, 
    required=True, 
    help='Specify whether to use RAG or just include file as prompt (yes, no).'
)
args = parser.parse_args()

if __name__ == "__main__":
    main(args.file_type, args.model, args.use_rag)