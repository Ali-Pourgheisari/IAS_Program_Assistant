from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

load_dotenv(Path(__file__).resolve().parent / ".env")
data_dir = Path(__file__).resolve().parent / "Data"
reader = SimpleDirectoryReader(input_dir=str(data_dir), required_exts=[".txt"])
documents = reader.load_data()

splitter = SentenceSplitter(
    chunk_size=128,
    chunk_overlap=20
)

nodes = splitter.get_nodes_from_documents(documents)
embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
index = VectorStoreIndex(nodes, embed_model=embed_model)


llm = OpenAI(model="gpt-4o-mini", temperature=0)
query_engine = index.as_query_engine(
    llm=llm,
    similarity_top_k=3,
)

response = query_engine.query(
    "How long do I have to return a purchase?"
)

print(response)
for source in response.source_nodes:
    print("Source:", source.node.metadata.get("file_name"))
    print("Text:", source.node.text)