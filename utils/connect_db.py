from typing import Tuple

from pymongo.collection import Collection
from pymongo.mongo_client import MongoClient


def connect(username: str, password: str) -> Tuple[MongoClient, Collection]:
    username = "rpcarvs"
    cluster = "cluster0.5l4gvzk.mongodb.net"
    connection_string = (
        f"mongodb+srv://{username}:{password}@{cluster}?retryWrites=true&w=majority"
    )
    client = MongoClient(connection_string)
    # Access your database and collection
    database = client["library_embeddings"]

    return client, database["embeddings"]
