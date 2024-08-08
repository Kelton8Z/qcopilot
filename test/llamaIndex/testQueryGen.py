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
node_parser = SentenceSplitter(chunk_size=1024)
nodes = node_parser.get_nodes_from_documents(documents)
'''
fileToID = {}
for node, doc in zip(nodes, documents):
    #node.metadata["doc_id"] = doc.doc_id
    filepath = node.metadata["file_path"]
    fileToID[filepath] = doc.node_id 
'''
nodeToFile = {}
for node in nodes:
    nodeToFile[node.node_id] = node.metadata["file_path"] 
    
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
queryToDoc = {query: nodeToFile[doc[0]] for query, doc in relevant_docs.items()}

### INDEXING
from llama_index.embeddings.openai import OpenAIEmbedding
embed_model = OpenAIEmbedding(model="text-embedding-3-large", api_base=openai_api_base)

index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", streaming=True)

### EVAL 
# from langchain_core.tracers.context import tracing_v2_enabled
from langsmith import Client, traceable
from langsmith.run_helpers import get_current_run_tree

os.environ["LANGCHAIN_PROJECT"] = "source"
os.environ["LANGCHAIN_TRACING_V2"] = "true" 
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
langchain_api_key = os.environ["LANGCHAIN_API_KEY"] = st.secrets.langsmith_key


#@traceable
def test():
    '''
    run = get_current_run_tree()
    run_id = str(run.id)
    '''

    top1hit = 0
    top3hit = 0
    total = len(queries)
    for queryID, query in queries.items():
        response = st.session_state.chat_engine.chat(query)
        response_msg = response.response
        source_nodes = response.source_nodes
        source_filepaths = [node.metadata["file_path"] for node in source_nodes]
        if source_nodes:
            groundtruth_doc = queryToDoc[queryID]
            top1node = source_nodes[0]
            top1_filepath = source_nodes[0].metadata["file_path"]
            top1_doc_id = top1node.node_id #fileToID[top1_filepath]
            #top3_doc_IDs = [fileToID[filepath] for filepath in source_filepaths[:3]]
            
            #assert(top1_doc_id in nodeToFile)
            #assert(groundtruth_doc in nodeToFile)
            top1hit += int(top1_filepath==groundtruth_doc)
            top3_docs = [node.metadata["file_path"] for node in source_nodes[:3]]
            
            top3hit += int(groundtruth_doc in top3_docs)
        
        import requests
        ''' 
        requests.patch(
            f"https://api.smith.langchain.com/runs/{run_id}",
            json={
                "inputs": {"text": query},
                "outputs": {"output": response_msg, "sources": source_filepaths},
            },
            headers={"x-api-key": langchain_api_key},
        )'''
            
        st.session_state.chat_engine.reset()

    top1HitRate = top1hit/total
    top3HitRate = top3hit/total
    print(f'{total} pairs')
    print(f'{top1HitRate}@1')
    print(f'{top3HitRate}@3')
        
test()
