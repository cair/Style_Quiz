import numpy as np
import pandas as pd

def leave_one_out_split(outfit_ids, groups, derived_booking_times):
    outfit_ids = np.array(outfit_ids)
    groups = np.array(groups)
    derived_booking_times = np.array(derived_booking_times)
    sorted_indices = np.argsort(derived_booking_times)
    return outfit_ids[sorted_indices[:-1]], outfit_ids[sorted_indices[-1]], groups[sorted_indices[:-1]], groups[sorted_indices[-1]], derived_booking_times[sorted_indices[:-1]], derived_booking_times[sorted_indices[-1]]

def leave_one_out_split_unique(outfit_ids, groups, derived_booking_times):
    outfit_ids = np.array(outfit_ids)
    groups = np.array(groups)
    derived_booking_times = np.array(derived_booking_times)
    
    sorted_indices = np.argsort(derived_booking_times)
    sorted_outfit_ids = outfit_ids[sorted_indices]
    sorted_groups = groups[sorted_indices]
    sorted_booking_times = derived_booking_times[sorted_indices]
    
    unique_groups, counts = np.unique(sorted_groups, return_counts=True)
    
    single_count_indices = np.where(counts == 1)[0]
    if len(single_count_indices) == 0:
        print(f"No unique outfit found with groups {groups}")
        return None
    
    unique_group = unique_groups[single_count_indices[0]]
    unique_group_index = np.where(sorted_groups == unique_group)[0][0]
    remaining_indices = np.arange(len(sorted_groups)) != unique_group_index
    
    return (
        sorted_outfit_ids[remaining_indices], sorted_outfit_ids[unique_group_index],
        sorted_groups[remaining_indices], sorted_groups[unique_group_index],
        sorted_booking_times[remaining_indices], sorted_booking_times[unique_group_index]
    )

def convert_user_orders_to_train_test_splits(user_orders_df):
    user_splits = user_orders_df.apply(lambda x: leave_one_out_split(x["outfit.id"], x["group"], x["derived.bookingTime"]), axis=1)
    user_splits_df = pd.DataFrame(user_splits.tolist(), columns=["train_outfit_ids", "test_outfit_id", "train_group", "test_group", "train_booking_times", "test_booking_time"])
    user_splits_unique = user_orders_df.apply(lambda x: leave_one_out_split_unique(x["outfit.id"], x["group"], x["derived.bookingTime"]), axis=1)
    user_splits_unique_df = pd.DataFrame(user_splits_unique.tolist(), columns=["train_outfit_ids", "test_outfit_id", "train_group", "test_group", "train_booking_times", "test_booking_time"])
    user_splits_unique_df = user_splits_unique_df.dropna()
    return user_splits_df, user_splits_unique_df