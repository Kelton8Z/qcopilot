import numpy as np
import streamlit as st
from llama_index.llms.azure_openai import AzureOpenAI
# from openai import AzureOpenAI
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
# from llama_index.core.chat_engine import SimpleChatEngine

azure_api_key = st.secrets.azure_api_key
azure_endpoint = st.secrets.azure_endpoint
azure_chat_deployment = st.secrets.azure_chat_deployment
azure_embedding_deployment = st.secrets.azure_embedding_deployment

api_version = "2024-03-01-preview"

llm = AzureOpenAI(
    # model="gpt-35-turbo-16k",
    azure_deployment=st.secrets.azure_chat_deployment,
    api_key=azure_api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
)

import logging
from typing import Any, List, Optional
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.callbacks import trace_method
from llama_index.core.chat_engine.types import (
    StreamingAgentChatResponse,
)
from llama_index.core.base.response.schema import (
    RESPONSE_TYPE,
    StreamingResponse,
    # AsyncStreamingResponse,
)
from llama_index.core.types import Thread
from llama_index.core.chat_engine.utils import (
    response_gen_from_query_engine,
    # aresponse_gen_from_query_engine,
)

logger = logging.getLogger(__name__)

@trace_method("chat")
def patched_stream_chat(
    self, message: str, chat_history: Optional[List[ChatMessage]] = None
) -> StreamingAgentChatResponse:
    chat_history = chat_history or self._memory.get(input=message)

    # Generate standalone question from conversation context and last message
    condensed_question = self._condense_question(chat_history, message)
    self.condensed_question = condensed_question

    log_str = f"Querying with: {condensed_question}"
    logger.info(log_str)
    if self._verbose:
        print(log_str)

    # TODO: right now, query engine uses class attribute to configure streaming,
    #       we are moving towards separate streaming and non-streaming methods.
    #       In the meanwhile, use this hack to toggle streaming.
    from llama_index.core.query_engine.retriever_query_engine import (
        RetrieverQueryEngine,
    )

    if isinstance(self._query_engine, RetrieverQueryEngine):
        is_streaming = self._query_engine._response_synthesizer._streaming
        self._query_engine._response_synthesizer._streaming = True

    # Query with standalone question
    query_response = self._query_engine.query(condensed_question)

    # NOTE: reset streaming flag
    if isinstance(self._query_engine, RetrieverQueryEngine):
        self._query_engine._response_synthesizer._streaming = is_streaming

    tool_output = self._get_tool_output_from_response(
        condensed_question, query_response
    )

    # Record response
    if (
        isinstance(query_response, StreamingResponse)
        and query_response.response_gen is not None
    ):
        # override the generator to include writing to chat history
        self._memory.put(ChatMessage(role=MessageRole.USER, content=message))
        response = StreamingAgentChatResponse(
            chat_stream=response_gen_from_query_engine(query_response.response_gen),
            sources=[tool_output],
        )
        thread = Thread(
            target=response.write_response_to_history,
            args=(self._memory,),
        )
        thread.start()
    else:
        raise ValueError("Streaming is not enabled. Please use chat() instead.")
    return response

'''
import sys
import os
grandparent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   
sys.path.append(grandparent_dir)     
from readFeishuWiki import CustomExcelReader
dir = "/Users/keltonzhang/Code/qc/data"
reader = SimpleDirectoryReader(
    input_dir=dir, 
    recursive=True, 
    filename_as_id=True,
    file_extractor={".xlsx": CustomExcelReader()}, 
    # required_exts=[".xlsx"],
    file_metadata=lambda filename: {"file_name": filename}
)
docs = reader.load_data()
print(f'indexed {len(docs)} docs')

embed_model = AzureOpenAIEmbedding(
    model="text-embedding-ada-002",
    deployment_name=azure_embedding_deployment,
    api_key=azure_api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
)
index = VectorStoreIndex.from_documents(
    docs, embed_model=embed_model
)
'''

query = "compare Vicuna and Llama3"
csv = '''
,0,1,2,3,4,5,6,7
0,Model,Input,Output,Throughput（requests/s),Throughput（tokens/s),Avg latency（Batch=1）,Avg latency（Batch=8）,Avg latency（Batch=64）
1,Aquila-7B,60,20,38.19,3055.14,636.76,771.71,1775.86
2,Aquila-7B,128,20,20.76,3072.22,642.72,882.7,2792.28
3,Aquila2-7B,60,20,38.2,3056.2,637.89,766.23,1171.65
4,Aquila2-7B,128,20,20.45,3025.99,645.13,886.26,2805.52
5,Baichuan-7B,60,20,39.81,3184.63,625.2,753.94,1723.33
6,Baichuan-7B,128,20,21.34,3158.29,629.15,871.17,2793.17
7,Baichuan2-7B,60,20,39.83,3186.09,648.05,"771,66",1790.73
8,Baichuan2-7B,128,20,19.88,2942.16,654.05,881.51,3579.45
9,Llama2-7B-hf,60,20,42.38,3390.27,611.84,740.53,1687.02
10,Llama2-7B-hf,128,20,22.13,3274.77,616.09,851.13,2751.64
11,Llama3-8B,60,20,47.96,3837.01,711.23,811.11,1759.25
12,Llama3-8B,128,20,24.15,3574.85,717.54,919.62,2687.6
13,Vicuna-7b,60,20,43.21,3456.9,530.74,736.39,1658.04
14,Vicuna-7b,128,20,23.07,3414.87,548.45,847.51,2753.68
'''

# query_engine = index.as_query_engine(llm=llm, response_mode="no_text")
# query_resp = query_engine.retrieve(query)
# print(query_resp)

'''
chat_engine = index.as_chat_engine(chat_mode="condense_question", llm=llm, streaming=True)
chat_engine.stream_chat = patched_stream_chat.__get__(chat_engine)
chat_engine._verbose = True

query_engine = index.as_query_engine(llm=llm)

outputs = []
for i in range(3):
    resp = query_engine.query(query)
    outputs.append(resp.response)
    # chat_engine.chat_history = []
    
print(outputs)
'''
from openai import OpenAI
client = OpenAI()
prompt = query+"\n"+csv
for i in range(2):
    resp = client.chat.completions.create(
    model="gpt-4",
    messages=[{
            "role": "user",
            "content": prompt,
            #   "attachments": [
            #     {
            #       "file_id": message_file.id,
            #       "tools": [{"type": "code_interpreter"}]
            #     }
            #   ]
            }]
    ,
    #   best_of=5,
    logprobs=True,
    top_logprobs=2
    #   temperature=0
    )
    # print(resp.choices)
    top_seq_logprobs = [choice.logprobs for choice in resp.choices]
    '''
    top_two_logprobs = resp.choices[0].logprobs.content[-1].top_logprobs
    content = ""
    for i, logprob in enumerate(top_two_logprobs, start=1):
        content += (
            f"output token {i}: {logprob.token}, "
            f"logprobs: {logprob.logprob}, "
            f"linear probability: {np.round(np.exp(logprob.logprob)*100,2)}%\n"
        )
    '''
    msg = ""
    for i, logprob in enumerate(top_seq_logprobs):
        content = logprob.content
        seq = "".join(tokenLogProb.token for tokenLogProb in content)
        msg += (
            f"seq {i+1}: {seq}, \n"
            f"linear probability: {np.round(np.exp(content[-1].logprob)*100,2)}%\n"
        )
    print(msg)
    
    print('---------------------------------------')


# print(chat_engine.condensed_question)
# '''

# completion = llm.chat.completions.create(
#     model="gpt-35-instant",  # e.g. gpt-35-instant
#     messages=[
#         {
#             "role": "user",
#             "content": query,
#         },
#     ],
#     logprobs=True,
# )

# choices = completion.choices
# content = choices[0].logprobs.content
# output = ""
# logprobs = content
# seq = ""
# for logprob in logprobs:
#     seq += logprob.token
# seq_prob = np.round(np.exp(logprobs[-1].logprob)*100, 2)
# output += f' {seq}, linear probability: {seq_prob}'
# print(output)

'''
# # Exponentiate the log probabilities
exp_probs = np.exp(log_probs)
# # Normalize to get the probabilities
probs = exp_probs / np.sum(exp_probs)
print(probs)
'''