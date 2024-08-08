import numpy as np
import streamlit as st
# from llama_index.llms.azure_openai import AzureOpenAI
from openai import AzureOpenAI

# from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
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

completion = llm.chat.completions.create(
    model="gpt-35-instant",  # e.g. gpt-35-instant
    messages=[
        {
            "role": "user",
            "content": "hi",
        },
    ],
    logprobs=True,
)

choices = completion.choices
content = choices[0].logprobs.content

'''
# # Exponentiate the log probabilities
exp_probs = np.exp(log_probs)
# # Normalize to get the probabilities
probs = exp_probs / np.sum(exp_probs)
print(probs)
'''

html_output = ""
logprobs = content
seq = ""
for logprob in logprobs:
    seq += logprob.token
seq_prob = np.round(np.exp(logprobs[-1].logprob)*100, 2)
html_output += f'<p style="color:cyan">has_sufficient_context_for_answer: {seq}, <span style="color:magenta">linear probability: {seq_prob}%</span></p>'
from IPython.display import display, HTML
display(HTML(html_output))
# print(completion.to_json())
# response = llm.chat(["hi"])
# chat_engine = SimpleChatEngine.from_defaults(llm=llm)
# response = chat_engine.chat("hi")
# print(dir(response))
# print(response.response)