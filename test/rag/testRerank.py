from llama_index.postprocessor.jinaai_rerank import JinaRerank

jina_rerank = JinaRerank(model="jina-reranker-v2-base-multilingual", api_key="<YOUR JINA AI API KEY HERE>", top_n=1)

# query_engine = index.as_query_engine(
#     similarity_top_k=10, llm=mixtral_llm, node_postprocessors=[jina_rerank]
# )