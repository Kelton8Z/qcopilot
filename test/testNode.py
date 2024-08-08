import os
import streamlit as st
from llama_index.core.evaluation import generate_question_context_pairs
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI 

directory = "./data"
openai_api_base = "http://vasi.chitu.ai/v1"
os.environ["OPENAI_API_BASE"] = openai_api_base
os.environ["OPENAI_API_KEY"] = st.secrets.openai_key

### QUERY GEN
llm = OpenAI(model="gpt-3.5-turbo-0125", max_tokens=50)
documents = SimpleDirectoryReader(directory).load_data()
index = VectorStoreIndex.from_documents(documents)
node_parser = SentenceSplitter(chunk_size=1024)
nodes = node_parser.get_nodes_from_documents(documents[:1])
print(f"Got nodes from document: {nodes}")
qa_dataset = generate_question_context_pairs(
    nodes, llm=llm, num_questions_per_chunk=1
)
queries = qa_dataset.queries

'''
fields of dataset e.g. 
queries={
    'acbf9e13-bbd6-4f72-bf60-cb149edefbe8': 'How does the information provided in the context contribute to the overall understanding of the topic being discussed?', 
    '163243d4-6bb4-410a-98b2-f59c993bb080': 'Can you identify any potential implications or applications of the information presented in the context?'
} 
corpus={
    '00b89bb7-ae7b-4e61-be54-ccb7587d74f7': 'blah'
} 
    
relevant_docs={'acbf9e13-bbd6-4f72-bf60-cb149edefbe8': ['00b89bb7-ae7b-4e61-be54-ccb7587d74f7'], 
'163243d4-6bb4-410a-98b2-f59c993bb080': ['00b89bb7-ae7b-4e61-be54-ccb7587d74f7']} 

mode='text'
'''

relevant_docs =  qa_dataset.relevant_docs
queryToDoc = {query: doc[0] for query, doc in relevant_docs.items()}

index = VectorStoreIndex.from_documents(documents)
st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", streaming=True)
query = queries[0]
response = st.session_state.chat_engine.chat(query)
response_msg = response.response
source_nodes = response.source_nodes
print(f'referenced {source_nodes}')