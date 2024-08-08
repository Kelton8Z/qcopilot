# from openai import OpenAI
# client = OpenAI()

# completion = client.chat.completions.create(
#   model="gpt-3.5-turbo",
#   messages=[
#     {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
#     {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
#   ]
# )

# print(completion.choices[0].message)
import os
import streamlit as st
from llama_index.llms.openai import OpenAI

os.environ["OPENAI_API_BASE"] = "http://vasi.chitu.ai/v1"
os.environ["OPENAI_API_KEY"] = st.secrets.openai_key
client = OpenAI(model="gpt-4o", api_key=st.secrets.openai_key)
print(client.complete("Paul Graham is "))

# from llama_index.embeddings.openai import OpenAIEmbedding
# embed_model = OpenAIEmbedding(model="text-embedding-3-large", api_key=st.secrets.claude_key)