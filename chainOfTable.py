import pandas as pd
from llama_index.core.llama_pack import download_llama_pack

download_llama_pack(
    "ChainOfTablePack",
    "./chain_of_table_pack",
    # leave the below line commented out if using the notebook on main
)
from chain_of_table_pack.llama_index.packs.tables.chain_of_table.base import ChainOfTableQueryEngine, serialize_table

from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-4-1106-preview")
df = pd.read_excel("./data/模型性能对比.xlsx")
query_engine = ChainOfTableQueryEngine(df, llm=llm, verbose=True)

query_engine.query("how many columns are there in the vllm table?")