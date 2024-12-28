from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import time

from backend.app.logger_config import setup_logger

logger = setup_logger(__name__)

class PineconeClient:
    def __init__(self, api_key):
        self.client = Pinecone(api_key=api_key)

    def create_index(self, index_name):
        if self.client.has_index(index_name):
            self.client.delete_index(index_name)

        self.client.create_index(
            name=index_name,
            dimension=1536, # dimensionality of text-embedding-ada-002
            metric="dotproduct",
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
        self.poll_for_index_creation_convergence(index_name)

    def poll_for_index_update_convergence(self, index_name, expected_total_count, max_retries=10, wait_time_secs=2):
        index = self.client.Index(index_name)

        retries = 0
        while retries < max_retries:
            stats = index.describe_index_stats()
            current_count = stats["total_vector_count"]

            if current_count >= expected_total_count:
                logger.info("Successfully updated index.")
                return True

            time.sleep(wait_time_secs)
            retries += 1

        logger.error(f"Error: index update did not converge to expected total record count: {expected_total_count}")
        return False

    def poll_for_index_creation_convergence(self, index_name):
        while not self.client.describe_index(index_name).status['ready']:
            time.sleep(1)
