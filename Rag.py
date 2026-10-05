from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import QueryEngineTool


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

shop_policy_tool = QueryEngineTool.from_defaults(
    query_engine=index.as_query_engine(
        llm=OpenAI(model="gpt-4o-mini", temperature=0),
        similarity_top_k=3,
    ),
    name="query_engine",
    description="A query engine for retrieving information about the company's policies.",
)



result = shop_policy_tool.call("What is the return window?")
print(result)