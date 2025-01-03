# Overview

An enterprise assistant application on top of a [Lark knowledge base](https://www.larksuite.com/en_us/product/wiki) featuring:
- LLM integration with gpt4o, claude3.5 and self-hosted models
- Streamlit app with file upload, answer feedback, model switching, starter questions UI
- RAG orchestrated by LlamaIndex with granular reference into Lark document blocks
- Eval with custom pipeline focused on testing data analysis queries
- Message logging with Langsmith generating metrics such as time to token, error rate
- Persist user uploaded file with S3
- File access control based on Lark user credential managment 
