import numpy as np
import pandas as pd

from src.collect_database_data import construct_user_orders, format_outfit_array, get_outfit_pictures
from src import load_dataframes
from resources.constants import *

def get_dataframes(load_local_dataframes=False, include_tag_data=True):
    if load_local_dataframes:
        if not os.path.isdir(DATA_SAVE_PATH):
            os.mkdir(DATA_SAVE_PATH)

        print("Retrieving order data from database...")
        orders_df = construct_user_orders()
        load_dataframes.save_pickle(orders_df, DATA_SAVE_PATH, ORDERS_PATH)

        print("Constructing outfit data...")
        user_rented_outfits = orders_df["outfit.id"].dropna().unique()
        outfits_df = format_outfit_array(user_rented_outfits, include_tag_data=include_tag_data, most_recent_instance=True)
        load_dataframes.save_pickle(outfits_df, DATA_SAVE_PATH, OUTFITS_PATH)

        print("Constructing outfit pictures...")
        pictures_df = get_outfit_pictures(outfits_df)
        load_dataframes.save_pickle(pictures_df, DATA_SAVE_PATH, PICTURES_PATH)

        print("Finished formatting downloading data.")
    else:
        orders_df = load_dataframes.load_pickle(DATA_SAVE_PATH, ORDERS_PATH)
        outfits_df = load_dataframes.load_pickle(DATA_SAVE_PATH, OUTFITS_PATH)
        pictures_df = load_dataframes.load_pickle(DATA_SAVE_PATH, PICTURES_PATH)
        print("Finished loading dataframes.")
    return orders_df, outfits_df, pictures_df

def build_dataset():
    orders_df, outfits_df, pictures_df = load_dataframes(load_dataframes=False)

    # Construct user activity triplets
    print(orders_df.shape, orders_df["customer.id"].nunique(), orders_df["outfit.id"].nunique())
    triplets_df = orders_df[USER_ACTIVITY_TRIPLET_COLUMNS].dropna()
    print(triplets_df.shape, triplets_df["customer.id"].nunique(), triplets_df["outfit.id"].nunique())

    customer_ids = triplets_df["customer.id"].unique()
    np.random.shuffle(customer_ids)
    customer_id_dict = {customer_id: i for i, customer_id in enumerate(customer_ids)}
    triplets_df["customer.id"] = triplets_df["customer.id"].apply(lambda x: customer_id_dict[x])

    triplets_df.to_csv(DATASET_FOLDER + "user_activity_triplets.csv", index=False)
    
    pass

if __name__ == "__main__":
    build_dataset()