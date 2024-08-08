from llama_index.core.evaluation import generate_question_context_pairs
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI 

llm = OpenAI(model="gpt-4o")
persist_dir = "chunks"

from llama_index.core import StorageContext, load_index_from_storage

try:
    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)

    # load index
    index = load_index_from_storage(storage_context)
except:
    documents = SimpleDirectoryReader("./data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    # index.storage_context.persist(persist_dir=persist_dir)

    node_parser = SentenceSplitter(chunk_size=512)
    nodes = node_parser.get_nodes_from_documents(documents)
    index.build_index_from_nodes(nodes)

nodes = index.docstore.docs.values()

qa_dataset = generate_question_context_pairs(
    nodes, llm=llm, num_questions_per_chunk=2
)
# from llama_index.core.schema import TextNode
# nodes = [TextNode(chunk) for chunk in chunks]

queries = qa_dataset.queries.values()
print(list(queries)[2])