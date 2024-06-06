import pandas as pd
from openai import OpenAI
import os

os.environ["OPENAI_API_BASE"] = ""
os.environ["OPENAI_API_KEY"] = ""
client = OpenAI()

excel_file_path = 'data/模型性能对比.xlsx'
'''
create_file_response = client.files.create(
    file=open(excel_file_path, 'rb'),
    purpose='assistants'
)
if create_file_response:
    file_id = create_file_response['id']
    print(f'openai file {file_id} created')
    content = client.files.content(file_id)
'''
assistant = client.beta.assistants.create(
  name="Financial Analyst Assistant",
  instructions="You are an expert ai infra analyst. Use your knowledge base to answer questions about ai model/hardware performance.",
  model="gpt-4o",
  tools=[{"type": "code_interpreter"},{"type": "file_search"}]
)

# Create a vector store caled "Financial Statements"
vector_store = client.beta.vector_stores.create(name="Model Performances")

# Ready the files for upload to OpenAI

df = pd.read_excel(excel_file_path, skiprows=1)
csv_file_path = excel_file_path.split('.xlsx')[0]+'.csv'
df.to_csv(csv_file_path, index=False)

'''
file_paths = [csv_file_path]
file_streams = [open(path, "rb") for path in file_paths]

# Use the upload and poll SDK helper to upload the files, add them to the vector store,
# and poll the status of the file batch for completion.
file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
  vector_store_id=vector_store.id, files=file_streams
)

# You can print the status and the file counts of the batch to see the result of this operation.
print(file_batch.status)
print(file_batch.file_counts)
'''
# Upload the user provided file to OpenAI
message_file = client.files.create(
  file=open(excel_file_path, "rb"), purpose="assistants"
)

print(f'provided {message_file.id} to openai')

'''
annotations = message_content.annotations
citations = []
for index, annotation in enumerate(annotations):
    message_content.value = message_content.value.replace(annotation.text, f"[{index}]")
    if file_citation := getattr(annotation, "file_citation", None):
        cited_file = client.files.retrieve(file_citation.file_id)
        citations.append(f"[{index}] {cited_file.filename}")

print(message_content.value)
print("\n".join(citations))
'''
assistant = client.beta.assistants.update(
  assistant_id=assistant.id,
  tool_resources={
      "file_search": {
          "vector_store_ids": [vector_store.id]
      },
      "code_interpreter": {
        "file_ids": [message_file.id]
      }
  },
)


def test_function_performance(model, input_len, output_len, throughput_requests, throughput_tokens, avg_latency_batch1, avg_latency_batch8, avg_latency_batch64):
    ch_question = f'{model}模型输入长度为{input_len}输出长度为{output_len}时吞吐量(request/s)，吞吐量(tokens/s)，batch是1、8、64的平均延迟分别是多少? 只返回最终答案数字[Throughput（requests/s), Throughput（tokens/s), Avg latency（Batch=1）, Avg latency（Batch=8）, Avg latency（Batch=64）]'
    en_question = f'With input leng {input_len}, output length {output_len}, what are the Throughput（requests/s), Throughput（tokens/s), Avg latency（Batch=1）, Avg latency（Batch=8）, Avg latency（Batch=64）for {model}? Only return the numbers'

    # Create a thread and attach the file to the message
    thread = client.beta.threads.create(
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


    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id, assistant_id=assistant.id
    )

    messages = list(client.beta.threads.messages.list(thread_id=thread.id, run_id=run.id))

    message_content = messages[0].content[0].text
    msg = message_content.value
    cur_hit = sum([str(throughput_requests) in msg, str(throughput_tokens) in msg, str(avg_latency_batch1) in msg, str(avg_latency_batch8) in msg, str(avg_latency_batch64) in msg])
    return cur_hit, 5-cur_hit

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
    cur_hit, cur_miss = test_function_performance(model, input_len, output_len, throughput_requests, throughput_tokens, avg_latency_batch1, avg_latency_batch8, avg_latency_batch64)
    hit += cur_hit
    miss += cur_miss

print(f'accuracy: {hit/(hit+miss)}')