import pandas as pd
from openai import OpenAI
import os
import streamlit as st

# os.environ["OPENAI_API_BASE"] = ""
os.environ["OPENAI_API_KEY"] = st.secrets.openai_key
openai_client = OpenAI(api_key=st.secrets.openai_key)


import anthropic

anthropic_client = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key=st.secrets.claude_key,
)

excel_file_path = 'sheets/模型性能对比.xlsx'
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


def test_function_performance(model_type, model, input_len, output_len, throughput_requests, throughput_tokens, avg_latency_batch1, avg_latency_batch8, avg_latency_batch64, md_table="", message_file=None):
    ch_question = f'{model}模型输入长度为{input_len}输出长度为{output_len}时吞吐量(request/s)，吞吐量(tokens/s)，batch是1、8、64的平均延迟分别是多少? 只返回最终答案数字[Throughput（requests/s), Throughput（tokens/s), Avg latency（Batch=1）, Avg latency（Batch=8）, Avg latency（Batch=64）]'
    en_question = f'With input leng {input_len}, output length {output_len}, what are the Throughput（requests/s), Throughput（tokens/s), Avg latency（Batch=1）, Avg latency（Batch=8）, Avg latency（Batch=64）for {model}? Only return the numbers'

    if model_type.startswith("gpt"):
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
    elif model_type=="claude3.5sonnet":
        try:
            message = anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1024,
                messages=[
                    {"role": "user", "content": md_table+"\n"+en_question}
                ]
            )
            msg = message.content[0].text
        except Exception as e:
            print(f"An error occurred: {e}")
    else:
        print("model not found")
        return None
        
    cur_hit = sum([str(throughput_requests) in msg, str(throughput_tokens) in msg, str(avg_latency_batch1) in msg, str(avg_latency_batch8) in msg, str(avg_latency_batch64) in msg])      
    return cur_hit, 5-cur_hit

def main(file_type, model_type):
  # Ready the files for upload to LLM 
  file_path = excel_file_path
  df = pd.read_excel(excel_file_path)
  md_table = ""
  message_file = None
  
  if file_type=="md":
      file_path = excel_file_path.split('.xlsx')[0]+'.md'
      md_table = df.to_markdown(tablefmt="grid")
      with open(file_path, 'w') as file:
          file.write(md_table)
  elif file_type=="csv":
      file_path = excel_file_path.split('.xlsx')[0]+'.csv'
      csv_table = df.to_csv()
      with open(file_path, 'w') as file:
          file.write(csv_table)
    
  if model_type.startswith("gpt"):
      global openai_assistant
      openai_assistant = openai_client.beta.assistants.create(
        name=" Analyst Assistant",
        instructions="You are an expert ai infra analyst. Use your knowledge base to answer questions about ai model/hardware performance.",
        model=model_type,
        tools=[{"type": "code_interpreter"},{"type": "file_search"}]
      )
      # Create a vector store
      vector_store = openai_client.beta.vector_stores.create(name="Model Performances")
      # Upload the user provided file to OpenAI
      message_file = openai_client.files.create(
        file=open(file_path, "rb"), purpose="assistants"
      )

      print(f'provided {message_file.id} to openai')

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
  for index, row in tqdm(df.iterrows()):
      model = row['Model']
      input_len = row['Input']
      output_len = row['Output']
      throughput_requests = row['Throughput（requests/s)']
      throughput_tokens = row['Throughput（tokens/s)']
      avg_latency_batch1 = row['Avg latency（Batch=1）']
      avg_latency_batch8 = row['Avg latency（Batch=8）']
      avg_latency_batch64 = row['Avg latency（Batch=64）']
      cur_hit, cur_miss = test_function_performance(model_type, model, input_len, output_len, throughput_requests, throughput_tokens, avg_latency_batch1, avg_latency_batch8, avg_latency_batch64, md_table, message_file)
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
    help='Specify the model to use (gpt4o, claude3.5sonnet).'
)
args = parser.parse_args()

if __name__ == "__main__":
    main(args.file_type, args.model)