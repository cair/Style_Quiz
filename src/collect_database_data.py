#Migrated from FREja
import os
import sys
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
from tqdm import tqdm
from mysql.connector import (connection)
from resources.database_settings import DB_SETTINGS
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from resources.constants import *
from src.retrieve_image_bucket_data import retrieve_picture_ids


def get_db_query(query, keep_columns=None):
    cnx = connection.MySQLConnection(**DB_SETTINGS)
    cursor = cnx.cursor(dictionary=True)
    cursor.execute(query)
    results = cursor.fetchall()
    df_db = pd.DataFrame([list(order_dict.values()) for order_dict in results], columns=list(results[0].keys()))
    if keep_columns is not None:
        return df_db[keep_columns]
    else:
        return df_db

def get_outfit_data():
    return get_db_query(OUTFITS_QUERY)

def get_tag_data():
    return get_db_query(TAG_QUERY)

def get_picture_data():
    return get_db_query(PICTURES_QUERY)

def get_outfit_tag_data():
    return get_db_query(OUTFIT_TAG_QUERY)

def get_outfit_tags(outfit_id, outfit_tag_df, tag_df):
    relevant_tags = outfit_tag_df[outfit_tag_df["outfitsId"] == outfit_id]
    outfit_tags = tag_df[tag_df["id"].isin(relevant_tags["tagsId"].values)]
    return outfit_tags

def apply_tags(outfit_id, column_name, outfit_tag_df, tag_df):
    outfit_tags = get_outfit_tags(outfit_id, outfit_tag_df, tag_df)
    return outfit_tags[column_name].values

def get_outfit_size(outfit_tags, outfit_categories):
    if "Size" in outfit_categories:
        return outfit_tags[np.where(outfit_categories == "Size")]
    else:
        return np.array(["None"])

def get_tagged_outfits():
    print("Retrieving outfit data from database...")
    outfit_df = get_outfit_data()
    outfit_tag_df = get_outfit_tag_data()
    tag_df = get_tag_data()
    print("Updating outfit data with tags...")
    outfit_df["outfit_tags"] = outfit_df.apply(lambda row: apply_tags(row.id, "tag", outfit_tag_df, tag_df), axis=1)
    outfit_df["tag_categories"] = outfit_df.apply(lambda row: apply_tags(row.id, "tagCategory", outfit_tag_df, tag_df), axis=1)
    outfit_df["Outfit_size"] = outfit_df.apply(lambda row: get_outfit_size(row.outfit_tags, row.tag_categories), axis=1)#.drop(outfit_df.columns[6:13], axis=1)
    print("Encoding tag vectors...")
    tag_encoder = MultiLabelBinarizer()
    tag_encoder.fit(outfit_df["outfit_tags"].values)
    outfit_df["tag_encoding"] = outfit_df["outfit_tags"].apply(lambda x: tag_encoder.transform(x.reshape(1, -1))[0])
    print("Finished updating outfit data.")

    return outfit_df

def get_outfit_pictures(outfit_df=None):
    print("Retrieving outfit data from database...")
    if outfit_df is None:
        outfit_df = get_outfit_data()
    pictures_df = get_picture_data()
    print(f"Retrieved {len(outfit_df)} outfits and {len(pictures_df)} pictures from the database")
    pictures_df = pd.DataFrame(pictures_df[pictures_df["owner"].isin(outfit_df["id"].values)])
    print(f"Filtered to {len(pictures_df)} pictures of the {len(outfit_df)} outfits")
    print("Checking if pictures exist in dataset...")
    pictures_dataset = retrieve_picture_ids()
    file_exists = lambda x: x in pictures_dataset
    pictures_df["exists_in_dataset"] = pictures_df.apply(lambda row: file_exists(row.id), axis=1)
    pictures_df = pictures_df[pictures_df["exists_in_dataset"] == True].drop(columns=["exists_in_dataset"])
    print("Finished updating pictures data.")
    return pictures_df

# Parameters here once had default values
def update_outfit_data(outfits_df_path, pictures_df_path):
    outfits_df = get_tagged_outfits()
    outfits_df.to_pickle(outfits_df_path)
    
    outfit_pictures_df = get_outfit_pictures()
    outfit_pictures_df.to_pickle(pictures_df_path)
    return outfits_df, outfit_pictures_df

def get_outfit_array_from_db(outfits_array, keep_columns=OUTFITS_DF_KEEP_COLUMNS, most_recent_instance=True):
    outfit_df = pd.DataFrame(outfits_array, columns=["id"])

    # Since all the outfits we are looking for don't have a valid meta.validTo date, we need to find the latest version of each outfit.
    ids_tuple = tuple(outfit_df["id"].dropna().unique())
    OUTFITS_EXTENDED_QUERY = f"""
    SELECT `Outfits`.*
    FROM `Outfits`
    JOIN (
        SELECT `id`, MAX(`meta.validTo`) as max_validTo
        FROM `Outfits`
        WHERE `id` IN {ids_tuple}
        GROUP BY `id`
    ) AS LatestOutfits
    ON `Outfits`.`id` = LatestOutfits.`id`"""
    # ON `Outfits`.`id` = LatestOutfits.`id` AND `Outfits`.`meta.validTo` = LatestOutfits.max_validTo
    # """
    if most_recent_instance:
        OUTFITS_EXTENDED_QUERY += " AND `Outfits`.`meta.validTo` = LatestOutfits.max_validTo"
    retrieved_outfits_df = get_db_query(OUTFITS_EXTENDED_QUERY)
    retrieved_outfits_df = retrieved_outfits_df[keep_columns]
    outfit_df = outfit_df.merge(retrieved_outfits_df, on="id", how="left")
    return outfit_df

def get_pictures_array_from_db(outfits_array, most_recent_instance=True):
    outfit_df = pd.DataFrame(outfits_array, columns=["id"])

    # Since all the outfits we are looking for don't have a valid meta.validTo date, we need to find the latest version of each outfit.
    ids_tuple = tuple(outfit_df["id"].dropna().unique())
    print(f"Retrieving pictures for {len(ids_tuple)} outfits from database...")
    OUTFITS_EXTENDED_QUERY = f"""
    SELECT `Pictures`.*
    FROM `Pictures`
    JOIN (
        SELECT `owner`, MAX(`meta.validTo`) as max_validTo
        FROM `Pictures`
        WHERE `owner` IN {ids_tuple}
        GROUP BY `owner`
    ) AS LatestOutfits
    ON `Pictures`.`owner` = LatestOutfits.`owner`"""
    # ON `Outfits`.`id` = LatestOutfits.`id` AND `Outfits`.`meta.validTo` = LatestOutfits.max_validTo
    # """
    if most_recent_instance:
        OUTFITS_EXTENDED_QUERY += " AND `Pictures`.`meta.validTo` = LatestOutfits.max_validTo"
    retrieved_pictures_df = get_db_query(OUTFITS_EXTENDED_QUERY)
    return retrieved_pictures_df

#Occasionally useful to construct outfit data from an array of existing outfit ids.
#Such as when we want to construct representations of outfits from user backlogs over longer periods of time.
def format_outfit_array(outfits_array, include_tag_data=True, most_recent_instance=True):
    tqdm.pandas()
    print("Retrieving outfit data from database...")
    outfit_df = get_outfit_array_from_db(outfits_array, most_recent_instance=most_recent_instance)

    outfit_tag_df = get_outfit_tag_data()
    if include_tag_data:
        tag_df = get_tag_data()
        print("Updating outfit data with tags...")
        outfit_df["outfit_tags"] = outfit_df.progress_apply(lambda row: apply_tags(row.id, "tag", outfit_tag_df, tag_df), axis=1)
        #outfit tags don't appear to be marked as outdated in the same way outfits are. though it's possible this will cause issues down the line.
        outfit_df["tag_categories"] = outfit_df.progress_apply(lambda row: apply_tags(row.id, "tagCategory", outfit_tag_df, tag_df), axis=1)
        outfit_df["Outfit_size"] = outfit_df.progress_apply(lambda row: get_outfit_size(row.outfit_tags, row.tag_categories), axis=1)#.drop(outfit_df.columns[6:13], axis=1)
        print("Encoding tag vectors...")
    print("Finished updating outfit data.")

    return outfit_df

def format_all_outfits(include_tag_data=True):
    tqdm.pandas()
    print("Retrieving outfit data from database...")
    retrieved_outfits_df = get_db_query(OUTFITS_QUERY)
    outfit_df = retrieved_outfits_df[OUTFITS_DF_KEEP_COLUMNS]

    outfit_tag_df = get_outfit_tag_data()
    if include_tag_data:
        tag_df = get_tag_data()
        print("Updating outfit data with tags...")
        outfit_df["outfit_tags"] = outfit_df.progress_apply(lambda row: apply_tags(row.id, "tag", outfit_tag_df, tag_df), axis=1)
        #outfit tags don't appear to be marked as outdated in the same way outfits are. though it's possible this will cause issues down the line.
        outfit_df["tag_categories"] = outfit_df.progress_apply(lambda row: apply_tags(row.id, "tagCategory", outfit_tag_df, tag_df), axis=1)
        outfit_df["Outfit_size"] = outfit_df.progress_apply(lambda row: get_outfit_size(row.outfit_tags, row.tag_categories), axis=1)#.drop(outfit_df.columns[6:13], axis=1)
        print("Encoding tag vectors...")
    print("Finished updating outfit data.")
    print(outfit_df.columns)
    return outfit_df

# Collect user order history for recommender system training data
def construct_user_orders():
    orders_df = get_db_query(USER_ORDER_QUERY, keep_columns=ORDER_KEEP_COLUMNS)
    rentals_df = get_db_query(SUBSCRIPTION_RENTALS_QUERY, keep_columns=RENTALS_KEEP_COLUMNS)
    orders_df = orders_df.merge(rentals_df, left_on="id", right_on="order", how="left")
    orders_df.rename(columns={"id_x": "id", "id_y":"subscription_id"}, inplace=True, errors="ignore")
    orders_df.dropna(subset=["subscription_id"], inplace=True)
    return orders_df

def construct_spot_orders():
    spot_rentals_df = get_db_query(SPOT_RENTALS_QUERY, keep_columns=SPOT_RENTALS_KEEP_COLUMNS)    
    return spot_rentals_df
if __name__ == "__main__":
    print("Debug main")