from resources.constants import *
from tqdm.notebook import tqdm
import numpy as np
import pandas as pd

def load_embeddings_form_folder():
    saved_embeddings = os.listdir(COMPUTED_EMBEDDINGS_PATH)
    embeddings_dict = {}
    for embedding in tqdm(saved_embeddings):
        embedding_name = ".".join(embedding.split(".")[:-1])
        embeddings_dict[embedding_name] = np.load(os.path.join(COMPUTED_EMBEDDINGS_PATH, embedding))
    return embeddings_dict

def load_embeddings_from_pickle():
    embeddings_df = pd.read_pickle(EMBEDDING_MODEL_PICKLE_PATH)
    embeddings_df = embeddings_df.groupby("outfit.id").agg({"picture.id": list, "embeddings": list}).reset_index()
    return embeddings_df

