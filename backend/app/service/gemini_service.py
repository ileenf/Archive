from backend.app.client.gemini_client import GeminiClient
from backend.app.logger_config import setup_logger

logger = setup_logger(__name__)

class GeminiService:
    def __init__(self, gemini_api_key):
        self.gemini_client = GeminiClient(gemini_api_key)

    def generate_response_with_context(self, top_docs, query):
        context = extract_page_content(top_docs)
        prompt = f"Given these journal entries: {context},\n answer this query: {query}.\n Provide exact journal entry dates only if specified, otherwise provide a general summary. Speak to me in 2nd person with 'you' statements."
        response = self.gemini_client.generate_response(prompt)

        return response.text

def extract_page_content(top_docs):
    return " ".join([doc.page_content for doc in top_docs])