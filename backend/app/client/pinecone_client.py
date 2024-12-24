from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import time


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

    def poll_for_index_update_convergence(self, index_name, new_count):
        index = self.client.Index(index_name)
        stats = index.describe_index_stats()
        current_count = stats["total_vector_count"]
        target_count = current_count + new_count

        while current_count != target_count:
            stats = index.describe_index_stats()
            current_count = stats["total_vector_count"]
            time.sleep(2)

    def poll_for_index_creation_convergence(self, index_name):
        while not self.client.describe_index(index_name).status['ready']:
            time.sleep(1)
