import numpy as np
import pandas as pd

def get_most_popular_outfits(df, n=10):
    most_popular_train_outfit_ids = df["train_outfit_ids"].explode().value_counts().index[:n]
    most_popular_train_groups = df["train_group"].explode().value_counts().index[:n]

    return np.array(most_popular_train_outfit_ids), np.array(most_popular_train_groups)

def get_previous_rentals(df, n=10):
    df["id_prediction"] = df["train_outfit_ids"].apply(lambda x: x if len(x) < n else x[n:])
    df["group_prediction"] = df["train_group"].apply(lambda x: x if len(x) < n else x[n:])

    return df

def pad_with_most_popular(x, pop_outfits, n=10):
    if len(x) < n:
        return np.append(x, pop_outfits[:n - len(x)])
    else:
        return x[:n]

def get_previous_rentals_pad_most_popular(df, n=10):
    most_popular_train_outfit_ids, most_popular_train_groups = get_most_popular_outfits(df, n)
    df["id_prediction"] = df.apply(lambda x: pad_with_most_popular(x["train_outfit_ids"], most_popular_train_outfit_ids, n), axis=1)
    df["group_prediction"] = df.apply(lambda x: pad_with_most_popular(x["train_group"], most_popular_train_groups, n), axis=1)
    return df