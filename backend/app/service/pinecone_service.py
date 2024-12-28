from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

from backend.app.client.pinecone_client import PineconeClient

class PineconeService:
    def __init__(self, index_name, pinecone_api_key, openai_api_key):
        self.pinecone_client = PineconeClient(pinecone_api_key)
        self.embeddings = OpenAIEmbeddings(model='text-embedding-ada-002', openai_api_key=openai_api_key)
        self.pinecone_client.create_index(index_name)
        self.vectorstore = PineconeVectorStore(index_name=index_name, embedding=self.embeddings)

    def add_documents_to_vectorstore(self, index_name):
        loader = TextLoader("journal_entries.txt")
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, separator=r"\s*Date:\s*", is_separator_regex=True)
        docs = text_splitter.split_documents(documents)

        self.vectorstore.add_documents(docs)
        self.pinecone_client.poll_for_index_update_convergence(index_name, len(docs))

    def generate_top_k_similar_documents(self, query, k):
        return self.vectorstore.similarity_search(query, k)