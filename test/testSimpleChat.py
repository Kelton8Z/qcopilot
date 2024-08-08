from llama_index.core.chat_engine import SimpleChatEngine

chat_engine = SimpleChatEngine.from_defaults()
prompt = "hello"
streaming_response = chat_engine.stream_chat(prompt)

response_msg = ""
for token in streaming_response.response_gen:
    response_msg += token
    
print(response_msg)