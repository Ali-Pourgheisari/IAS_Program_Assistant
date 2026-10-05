"""Build the retrieval index for the shop's policy documents."""

from pathlib import Path

from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"


def build_policy_index(data_dir: Path = DATA_DIR) -> VectorStoreIndex:
    """Load policy documents and index them using OpenAI embeddings."""
    load_dotenv(BASE_DIR / ".env")
    reader = SimpleDirectoryReader(input_dir=str(data_dir), required_exts=[".txt"])
    documents = reader.load_data()
    splitter = SentenceSplitter(chunk_size=128, chunk_overlap=20)
    nodes = splitter.get_nodes_from_documents(documents)
    embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
    return VectorStoreIndex(nodes, embed_model=embed_model)
