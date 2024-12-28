from fastapi import FastAPI
from dotenv import load_dotenv
import os

from backend.app.logger_config import setup_logger
from backend.app.service.pinecone_service import PineconeService

from backend.app.service.gemini_service import GeminiService

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
INDEX_NAME = "archive-index"
QUERY = "Tell me about the some times I went rock climbing."

app = FastAPI()
logger = setup_logger(__name__)
K = 4

@app.get("/")
async def root():
    pinecone_service = PineconeService(INDEX_NAME, PINECONE_API_KEY, OPENAI_API_KEY)
    pinecone_service.add_documents_to_vectorstore(INDEX_NAME)
    top_k_docs = pinecone_service.generate_top_k_similar_documents(QUERY, K)
    logger.info(f"Generated top {K} similar documents.")

    gemini_service = GeminiService(GEMINI_API_KEY)
    text_response = gemini_service.generate_response_with_context(top_k_docs, QUERY)
    logger.info("Generated LLM response with context.")

    return {"message": text_response}