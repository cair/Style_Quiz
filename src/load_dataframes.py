import os
import pandas as pd
from functools import lru_cache, _lru_cache_wrapper
import pickle
import gc

from resources.constants import *

def save_pickle(to_pickle_object, save_path, filename):
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    path = os.path.join(save_path, filename)
    if type(to_pickle_object) == pd.DataFrame:
        to_pickle_object.to_pickle(path)
    else:
        with open(path, "wb") as f:
            pickle.dump(to_pickle_object, f)

def load_pickle(dataframe_folder, dataframe_name, load_type="dataframe"):
    if not os.path.exists(dataframe_folder):
        os.makedirs(dataframe_folder)

    dataframe_path = os.path.join(dataframe_folder, dataframe_name)
    if not os.path.exists(dataframe_path):
        raise FileNotFoundError(f"File {dataframe_path} not found.")
    if load_type == "dataframe":
        return pd.read_pickle(dataframe_path)
    else:
        with open(dataframe_path, "rb") as f:
            loaded_pickle = pickle.load(f)
        return loaded_pickle


#Non-implemented functionality for backing up dataframes to GCS

#from src.api_resources_bucket_manager import get_bucket, download_from_gcs, upload_to_gcs
# def save_pickle(to_pickle_object, save_path, filename):
#     if not os.path.isdir(save_path):
#         os.makedirs(save_path)
#     path = os.path.join(save_path, filename)
#     if type(to_pickle_object) == pd.DataFrame:
#         to_pickle_object.to_pickle(path)
#     else:
#         with open(path, "wb") as f:
#             pickle.dump(to_pickle_object, f)
#     bucket = get_bucket()
#     upload_to_gcs(path, bucket, destination_path=path)

# def load_pickle(dataframe_folder, dataframe_name, load_type="dataframe"):
#     if not os.path.exists(dataframe_folder):
#         os.makedirs(dataframe_folder)

#     dataframe_path = os.path.join(dataframe_folder, dataframe_name)
#     if not os.path.exists(dataframe_path):
#         print(f"Downloading {dataframe_path} from GCS.")
#         bucket = get_bucket()
#         download_from_gcs(dataframe_path, dataframe_path, bucket)
#     if load_type == "dataframe":
#         return pd.read_pickle(dataframe_path)
#     else:
#         with open(dataframe_path, "rb") as f:
#             loaded_pickle = pickle.load(f)
#         return loaded_pickle

@lru_cache(maxsize = None)
def load_predictions(dataframe_folder):
    return load_pickle(dataframe_folder, PREDICTIONS_PATH)

@lru_cache(maxsize = None)
def load_outfits(dataframe_folder):
    return load_pickle(dataframe_folder, OUTFITS_PATH)

@lru_cache(maxsize = None)
def load_orders(dataframe_folder):
    return load_pickle(dataframe_folder, ORDERS_PATH)

@lru_cache(maxsize = None)
def load_pictures(dataframe_folder):
    return load_pickle(dataframe_folder, PICTURES_PATH)

def clear_caches():
    gc.collect()
  
    objects = [i for i in gc.get_objects() 
            if isinstance(i, _lru_cache_wrapper)]
    
    for object in objects:
        print(f"Clearing cache for {object}")
        print(f"Cache info: {object.cache_info()}")
        object.cache_clear()