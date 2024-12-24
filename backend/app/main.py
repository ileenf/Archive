from contextlib import asynccontextmanager

from fastapi import FastAPI
from dotenv import load_dotenv
import os
from app.service.pinecone_service import PineconeService

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_NAME = "archive-index"

app = FastAPI()

@app.get("/")
async def root():
    pinecone_service = PineconeService(INDEX_NAME, PINECONE_API_KEY, OPENAI_API_KEY)
    pinecone_service.add_documents_to_vectorstore(INDEX_NAME)
    top_k_docs = pinecone_service.generate_top_k_similar_documents("Info about highlights of living in Boston.", 4)
    return {"message": top_k_docs}