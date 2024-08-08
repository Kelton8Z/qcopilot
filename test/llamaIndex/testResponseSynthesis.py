from llama_index.core import get_response_synthesizer
import streamlit as st
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

api_version="2024-02-01"
azure_api_key = st.secrets.azure_api_key
azure_endpoint = st.secrets.azure_endpoint
azure_chat_deployment = st.secrets.azure_chat_deployment
azure_embedding_deployment = st.secrets.azure_embedding_deployment

llm = AzureOpenAI(
    model="gpt-35-turbo-16k",
    deployment_name=st.secrets.azure_chat_deployment,
    api_key=azure_api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
)

response_synthesizer = get_response_synthesizer(llm=llm, response_mode="compact")

import asyncio
import sys
import os
grandparent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   
sys.path.append(grandparent_dir)     
from readFeishuWiki import CustomExcelReader, readDocs
dir = "/Users/keltonzhang/Code/qc/data"
reader = SimpleDirectoryReader(
    input_dir=dir, 
    recursive=True, 
    filename_as_id=True,
    file_extractor={".xlsx": CustomExcelReader()}, 
    # required_exts=[".xlsx"],
    file_metadata=lambda filename: {"file_name": filename}
)
docs = reader.load_data()
print(f'indexed {len(docs)} docs')

embed_model = AzureOpenAIEmbedding(
    model="text-embedding-ada-002",
    deployment_name=azure_embedding_deployment,
    api_key=azure_api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
)
index = VectorStoreIndex.from_documents(
    docs, embed_model=embed_model
)

# index, fileToTitleAndUrl = asyncio.run(readDocs(docs, embed_model)) 

query = "compare Vicuna and Llama3"
chat_engine = index.as_chat_engine(chat_mode="condense_question", llm=llm, streaming=True)
# response = chat_engine.stream_chat(query)
# print(response.source_nodes)
# response_msg = ""
# for token in response.response_gen:
#     response_msg += token
# print(f'response {response_msg}')

from llama_index.retrievers.bm25 import BM25Retriever
import Stemmer

from llama_index.core.node_parser import SentenceSplitter

# initialize node parser
splitter = SentenceSplitter(chunk_size=512)

nodes = splitter.get_nodes_from_documents(docs)
# We can pass in the index, doctore, or list of nodes to create the retriever
bm25_retriever = BM25Retriever.from_defaults(
    nodes=nodes,
    similarity_top_k=5,
    # Optional: We can pass in the stemmer and set the language for stopwords
    # This is important for removing stopwords and stemming the query + text
    # The default is english for both
    stemmer=Stemmer.Stemmer("english"),
    language="english",
)
bm25_nodes = bm25_retriever.retrieve(query)
print(bm25_nodes)
print(len(bm25_nodes))

# import nest_asyncio

# nest_asyncio.apply()

from llama_index.core.retrievers import QueryFusionRetriever

retriever = QueryFusionRetriever(
    [
        index.as_retriever(similarity_top_k=5),
        BM25Retriever.from_defaults(
            docstore=index.docstore, similarity_top_k=5
        ),
    ],
    num_queries=1,
    use_async=False,
)

hybrid_nodes = retriever.retrieve(query)
print(hybrid_nodes)
print(len(hybrid_nodes))
processor = SimilarityPostprocessor(similarity_cutoff=0.25)

# query_engine = index.as_query_engine(response_synthesizer=response_synthesizer)
# response = query_engine.query(query)
# source_nodes = response.source_nodes
# print(f'all sources {source_nodes}')
filtered_nodes = processor.postprocess_nodes(hybrid_nodes)
print(f'filtered sources {filtered_nodes}')

response = response_synthesizer.synthesize(
    query, nodes=filtered_nodes
)
print(response)
