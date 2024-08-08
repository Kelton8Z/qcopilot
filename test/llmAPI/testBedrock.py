import streamlit as st
import requests
import json

from anthropic import AnthropicBedrock

client = AnthropicBedrock(
    aws_access_key=st.secrets.aws_access_key,
    aws_secret_key=st.secrets.aws_secret_key,
    aws_region="us-east-1",
)

message = client.messages.create(
    model="anthropic.claude-3-5-sonnet-20240620-v1:0",
    max_tokens=256,
    messages=[{"role": "user", "content": "Hello, world"}]
)
print(message.content)