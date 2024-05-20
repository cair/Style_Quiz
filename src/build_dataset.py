import numpy as np
import pandas as pd

from src.collect_database_data import construct_user_orders, format_outfit_array, format_all_outfits, get_outfit_pictures, construct_spot_orders
from src import load_dataframes
from resources.constants import *

def get_dataframes(load_local_dataframes=False, include_tag_data=True):
    if not load_local_dataframes:
        if not os.path.isdir(DATA_SAVE_PATH):
            os.mkdir(DATA_SAVE_PATH)

        print("Retrieving order data from database...")
        orders_df = construct_user_orders()
        load_dataframes.save_pickle(orders_df, DATA_SAVE_PATH, ORDERS_PATH)

        print("Retrieving spot order data from database...")
        spot_orders_df = construct_spot_orders()
        load_dataframes.save_pickle(spot_orders_df, DATA_SAVE_PATH, SPOT_RENTALS_PATH)
        
        print("Constructing outfit data...")
        user_rented_outfits = orders_df["outfit.id"].dropna().unique()
        #outfits_df = format_outfit_array(user_rented_outfits, include_tag_data=include_tag_data, most_recent_instance=True)
        outfits_df = format_all_outfits(include_tag_data=include_tag_data)
        outfits_df = outfits_df.sort_values(["id", "meta.validTo"], ascending=False).drop_duplicates("id")
        load_dataframes.save_pickle(outfits_df, DATA_SAVE_PATH, OUTFITS_PATH)

        print("Constructing outfit pictures...")
        pictures_df = get_outfit_pictures(outfits_df)
        load_dataframes.save_pickle(pictures_df, DATA_SAVE_PATH, PICTURES_PATH)

        print("Finished formatting downloading data.")
    else:
        orders_df = load_dataframes.load_pickle(DATA_SAVE_PATH, ORDERS_PATH)
        outfits_df = load_dataframes.load_pickle(DATA_SAVE_PATH, OUTFITS_PATH)
        pictures_df = load_dataframes.load_pickle(DATA_SAVE_PATH, PICTURES_PATH)
        spot_orders_df = load_dataframes.load_pickle(DATA_SAVE_PATH, SPOT_RENTALS_PATH)
        print("Finished loading dataframes.")
    return orders_df, spot_orders_df, outfits_df, pictures_df

def replace_tag_in_list(tag_list, tag_to_replace, replacement_tag):
    return [replacement_tag if tag == tag_to_replace else tag for tag in tag_list]

def build_dataset(load_local_dataframes=False):
    orders_df, spot_orders_df, outfits_df, pictures_df = get_dataframes(load_local_dataframes=load_local_dataframes)

    # Construct user activity triplets
    triplets_df = orders_df[USER_ACTIVITY_TRIPLET_COLUMNS].dropna()
    # Anonymize customer ids
    customer_ids = triplets_df["customer.id"].unique()
    np.random.shuffle(customer_ids)
    customer_id_dict = {customer_id: i for i, customer_id in enumerate(customer_ids)}
    triplets_df["customer.id"] = triplets_df["customer.id"].apply(lambda x: customer_id_dict[x])
    # Save to file
    triplets_df.to_csv(USER_ACTIVITY_TRIPLETS_CSV_PATH, index=False, sep=CSV_SEPARATOR)
    print("Finished constructing user activity triplets.")

    spot_triplets_df = spot_orders_df[SPOT_RENTALS_TRIPLET_COLUMNS]
    spot_triplets_df = spot_triplets_df.rename(columns=SPOT_RENTALS_RENAME_COLUMNS)
    spot_triplets_df.to_csv(SPOT_RENTALS_CSV_PATH, index=False, sep=CSV_SEPARATOR)
    print("Finished constructing spot rentals.")

    # Construct outfit data
    output_outfits_df = outfits_df.drop(["meta.validTo", "Outfit_size"], axis=1)
    output_outfits_df["description"] = output_outfits_df["description"].apply(lambda x: x.replace(";", ":") if x is not None else x) # Replace semicolons in descriptions with colons to avoid issues with csv
    output_outfits_df["outfit_tags"] = output_outfits_df["outfit_tags"].apply(list) # Convert these columns to lists to ensure they can be easily read via eval from the csv
    output_outfits_df["tag_categories"] = output_outfits_df["tag_categories"].apply(list) # Convert these columns to lists to ensure they can be easily read via eval from the csv
    output_outfits_df["tag_categories"] = output_outfits_df["tag_categories"].apply(lambda x: replace_tag_in_list(x, "brand", "Brand"))
    # Anonymize owner ids
    owner_ids = output_outfits_df["owner"].unique()
    np.random.shuffle(owner_ids)
    owner_id_dict = {owner_id: i for i, owner_id in enumerate(owner_ids)}
    output_outfits_df["owner"] = output_outfits_df["owner"].apply(lambda x: owner_id_dict[x])
    # Save to file
    output_outfits_df.to_csv(OUTFITS_CSV_PATH, index=False, sep=CSV_SEPARATOR)
    print("Finished constructing outfit data.")

    # Collect outfit pictures
    picture_triplets_df = pictures_df[["id", "owner", "displayOrder"]]
    picture_triplets_df.columns = ["picture.id", "outfit.id", "displayOrder"]
    picture_triplets_df = picture_triplets_df[picture_triplets_df["outfit.id"].isin(output_outfits_df["id"])].copy()
    picture_triplets_df.to_csv(PICTURE_TRIPLETS_CSV_PATH, index=False, sep=CSV_SEPARATOR)
    print("Finished constructing picture triplets.")

    # Third Chance outfit data
    third_chance_df = pd.read_excel(THIRD_CHANCE_RAW_PATH)
    
    def column_name_assignment(column_name):
        return column_name.lower().replace(" ", "_")

    def convert_date_added(date_string):
        if date_string == "-":
            return None
        else:
            return pd.to_datetime(date_string, format="%d-%m-%Y")

    third_chance_df = third_chance_df.drop(columns=THIRD_CHANCE_DROP_COLUMNS, errors="ignore")
    third_chance_df = third_chance_df.rename(mapper=column_name_assignment, axis=1) # Rename columns to lowercase, makes the format more consistent with the other dataframes
    third_chance_df = third_chance_df.dropna(subset=["name", "owner", "brand"], how="all") # Drop rows witout any identifying information
    third_chance_df["id"] = third_chance_df["id"].apply(lambda x: x if x != "-" else None)
    third_chance_df = third_chance_df.dropna(subset=["id"], how="any") # Drop all rows without an id
    third_chance_df = third_chance_df.dropna(subset=["condition", "retail_price", "tc_price"], how="all") # Drop rows without any pricing or condition information
    third_chance_df["date_added"] = third_chance_df["date_added"].apply(convert_date_added) # dates have an irregular format, convert it before saving to resolve the issue
    third_chance_df.to_csv(THIRD_CHANCE_CSV_PATH, index=False, sep=CSV_SEPARATOR)
    print("Finished constructing third chance outfit data.")

    return triplets_df, spot_triplets_df, output_outfits_df, picture_triplets_df

if __name__ == "__main__":
    build_dataset(load_local_dataframes=True)