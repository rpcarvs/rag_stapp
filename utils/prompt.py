from typing import Dict, List, Union

from streamlit import cache_data

from .connect_db import connect
from .embedding import get_embedding


@cache_data(ttl=600)
def get_query_results(
    query: str,
    username: str,
    password: str,
    limit: int = 5,
) -> List[Dict]:
    """
    1. transform query to embedded vector
    2. semantic search for similar vectors
    3. bring up the top 'limit' quotes
    """
    client, collection = connect(username=username, password=password)
    query_embedding = get_embedding(query)

    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "queryVector": query_embedding,
                "path": "embeddings",
                "exact": True,
                "limit": limit,
            }
        },
        {
            "$project": {
                "_id": 0,
                "text_chunk": 1,
                # "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]

    results = collection.aggregate(pipeline)
    client.close()

    return [doc for doc in results]


def prepare_prompt(question: str, documents: Union[List[Dict], None] = None) -> str:
    """
    1. run the query
    2. aggregate the returned info/quotes
    3. prepare prompt
    """
    context = ""
    if documents:
        text_documents = ""
        for doc in documents:
            text = doc.get("text_chunk", "")
            string = f"Quotes: {text}. \n"
            text_documents += string
        context = f"""Use the following pieces of context to
        improve the answer for the question at the end.
        {text_documents}"""

    prompt = context + f"Question: {question}"

    return prompt
