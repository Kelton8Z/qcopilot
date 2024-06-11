from llama_index.core.evaluation import generate_question_context_pairs
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI 

llm = OpenAI(model="gpt-4o")

documents = SimpleDirectoryReader("./data").load_data()
node_parser = SentenceSplitter(chunk_size=512)
nodes = node_parser.get_nodes_from_documents(documents)

qa_dataset = generate_question_context_pairs(
    nodes, llm=llm, num_questions_per_chunk=2
)
# from llama_index.core.schema import TextNode
# nodes = [TextNode(chunk) for chunk in chunks]

queries = qa_dataset.queries.values()
print(list(queries)[2])