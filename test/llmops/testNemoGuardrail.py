import pandas as pd
import streamlit as st
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS
import logging
from typing import Optional
from langchain.llms.base import BaseLLM
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from langchain_openai import AzureChatOpenAI
from nemoguardrails.actions.llm.utils import llm_call
from nemoguardrails.llm.params import llm_params
from nemoguardrails.llm.taskmanager import LLMTaskManager 
from nemoguardrails.llm.types import Task
from nemoguardrails import LLMRails, RailsConfig

api_version="2024-02-01"
azure_api_key = st.secrets.azure_api_key
azure_endpoint = st.secrets.azure_endpoint
azure_chat_deployment = st.secrets.azure_chat_deployment
azure_embedding_deployment = st.secrets.azure_embedding_deployment

import os
os.environ["AZURE_OPENAI_API_KEY"] = azure_api_key 
os.environ["AZURE_OPENAI_ENDPOINT"] = azure_endpoint
os.environ["AZURE_OPENAI_API_VERSION"] = api_version
os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"] = azure_chat_deployment
llm = AzureChatOpenAI(model="gpt-3.5-turbo", 
    azure_deployment=azure_chat_deployment,
    api_key=azure_api_key,  
    openai_api_version=api_version,
    azure_endpoint = azure_endpoint,
)
from openai import AzureOpenAI
azure_openai = AzureOpenAI( 
    azure_deployment=azure_chat_deployment,
    api_key=azure_api_key,  
    api_version=api_version,
    azure_endpoint = azure_endpoint,
)

azure_embedding_deployment = st.secrets.azure_embedding_deployment

embed_model = AzureOpenAIEmbedding(
    model="text-embedding-ada-002",
    deployment_name=azure_embedding_deployment,
    api_key=azure_api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
)

async def check_facts(evidence_list:list,
    llm_task_manager: LLMTaskManager,
    context: Optional[dict] = None,
    llm: Optional[BaseLLM] = None,
    
):
    """Checks the facts for the bot response."""

    evidence = evidence_list 
    response = context.get("last_bot_message")
    print('********')
    print(evidence)
    print('********')
    if evidence:
        prompt = llm_task_manager.render_task_prompt(
            task=Task.SELF_CHECK_FACTS,
            context={
                "evidence": evidence,
                "response": response,
            },
        )

        with llm_params(llm, temperature=0.0):
            entails = await llm_call(llm, prompt)

        entails = entails.lower().strip()
        logging.info(f"Entailment result is {entails}.")

        return "yes" in entails

    # If there was no evidence, we always return true
    return True

def create_index(documents, embedding_model):
    ann_index = FAISS.from_documents(documents=documents, embedding=embedding_model)
    return ann_index

directory = "/Users/keltonzhang/Code/qc/a"
from llama_index.core import SimpleDirectoryReader
reader = SimpleDirectoryReader(
    input_dir=directory,
    recursive=True,
    filename_as_id=True,
    file_metadata=lambda filename: {"file_name": filename},
    raise_on_error=True
)
docs = reader.load_data()
import faiss
from llama_index.core import (
    SimpleDirectoryReader,
    load_index_from_storage,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.vector_stores.faiss import FaissVectorStore
from IPython.display import Markdown, display
# dimensions of text-ada-embedding-002
# d = 1536
# faiss_index = faiss.IndexFlatL2(d)
# vector_store = FaissVectorStore(faiss_index=faiss_index)
# storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    docs#, storage_context=storage_context
)
# index = create_index(documents=docs, embedding_model=embed_model)

async def retrieve(query: str) -> list:
    # get relevant contexts 
    query_engine = index.as_query_engine(llm=llm)
    res = query_engine.query(query)
    print(res.source_nodes)
    contexts = [doc.text for doc in res.source_nodes]
    return contexts

async def rag(query: str, contexts: list) -> str:
    print("> Retrieval Augmented Generation Called\n")  # we'll add this so we can see when this is being used
    context_str = "\n".join(contexts)
    #print(context_str)
    # place query and contexts into RAG prompt
    prompt = f"""You are a helpful assistant, below is a query from a user and
    some relevant contexts. Answer the question given the information in those
    contexts. If you cannot find the answer to the question, say "I don't know".

    Contexts:
    {context_str}

    Query: {query}

    Answer: """
    res = azure_openai.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}])
    choices = res.choices
    top_choice = choices[0]
    print(top_choice.message.content)
    return top_choice.message.content

colang_content = """

# Handle Profanity
define user express_insult
  "You are stupid" 
  "I will shoot you"

define bot express_calmly_willingness_to_help
  "I won't engage with harmful content."
  
define flow handle_insult
  user express_insult
  bot express_calmly_willingness_to_help

# QA FLOW
define user ask_question
    "user ask a question"
    "what is salary?"
    "user ask about capital"
    "user ask about sql"

define flow handle_general_input
    user ask_question
    $contexts = execute retrieve(query=$last_user_message)
    $answer = execute rag(query=$last_user_message, contexts=$contexts)
    bot $answer
    $accurate = execute check_facts(evidence_list = $contexts)
    if not $accurate:
        bot remove last message
        bot inform answer unknown
define bot remove last message
    "(remove last message)"

"""

config = RailsConfig.from_content(colang_content=colang_content)
rails = LLMRails(config, llm=llm, verbose=False)
rails.register_action(action=retrieve, name="retrieve")
rails.register_action(action=rag, name="rag")
rails.register_action(action=check_facts, name="check_facts")

async def main():
    response = await rails.generate_async(prompt="What is capital of China?")
    print(response)

import asyncio
asyncio.run(main())