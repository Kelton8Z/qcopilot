import os
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.llms.openai import OpenAI

llama3_api_base = "http://localhost:25121/v1"
os.environ["OPENAI_API_BASE"] = llama3_api_base
llm = OpenAI(api_base=llama3_api_base, api_key="aba")

chat_engine = SimpleChatEngine.from_defaults(llm=llm)

resp = chat_engine.chat("who made you")
print(resp)