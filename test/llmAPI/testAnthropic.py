import anthropic
import streamlit as st

client = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key=st.secrets.claude_key,
)

opus = "claude-3-opus-20240229"
sonnet = "claude-3-5-sonnet-20240620"
message = client.messages.create(
    model=opus,
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Hello, Claude"}
    ]
)
print(message.content[0].text)
