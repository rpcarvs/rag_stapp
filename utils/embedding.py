# Load the embedding model
import os

from sentence_transformers import SentenceTransformer

# loading the model
os.environ["TOKENIZERS_PARALLELISM"] = "false"
path = os.path.join(os.path.dirname(__file__), "embs_model")
model_embeddings = SentenceTransformer(path)


# Define a function to generate embeddings
def get_embedding(data, precision="float32"):
    return model_embeddings.encode(data, precision=precision).tolist()
