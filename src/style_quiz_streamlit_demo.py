import os
os.chdir(r"C:\Users\kaborg15\Python_projects\Vibrent_Style_Quiz_Generation")

import streamlit as st
import time
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

from src.display_images import display_image_ids
import random

OUTFIT_EMBEDDINGS_DF_PATH = r"resources\data\outfit_embeddings_df.pkl"
REPRESENTATION_COLUMN = "outfit_embeddings"
NUM_SAMPLES_PER_CLUSTER = 27

# Note: not implemented for more than 2 clusters yet
NUM_CLUSTERS = 2

def cluster_current_split(current_cluster_df):
    outfit_representations = np.stack(current_cluster_df[REPRESENTATION_COLUMN].values)
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=0, n_init="auto").fit(outfit_representations)
    current_cluster_df["cluster"] = kmeans.labels_

    cluster_samples = []
    for i in range(NUM_CLUSTERS):
        cluster_outfits = current_cluster_df[current_cluster_df["cluster"] == i].copy()
        cluster_embeddings = np.stack(cluster_outfits[REPRESENTATION_COLUMN].values)
        num_cluster_samples = min(NUM_SAMPLES_PER_CLUSTER, len(cluster_outfits))
        
        # Properly represent the diversity of the cluster by applying KMeans to the embeddings
        cluster_kmeans = KMeans(n_clusters=num_cluster_samples, random_state=1, n_init="auto").fit(cluster_embeddings)
        cluster_outfits["representation_cluster"] = cluster_kmeans.labels_
        cluster_representation = cluster_outfits.groupby("representation_cluster").first().reset_index()
        cluster_samples.append(cluster_representation)
        
        st.session_state.text_widgets[i].write(f"Cluster {i}: {len(cluster_outfits)} outfits")

    return current_cluster_df, cluster_samples

def construct_cluster_collages(cluster_samples):
    print("Constructing cluster collages, number of clusters: ", len(cluster_samples))
    for i, cluster in enumerate(cluster_samples):
        image_ids = [st.session_state.outfits_to_lead_picture_id_dict[outfit_id] for outfit_id in cluster["id"].values]
        cluster_collage = display_image_ids(image_ids) 
        st.session_state.image_widgets[i].image(cluster_collage, caption=f"Cluster {i}")


def handle_radio_choice():
    print("Handling radio choice")
    chosen_cluster = st.session_state.chosen_cluster
    st.session_state.chosen_cluster = None
    if chosen_cluster == "Quit":
        raise SystemExit(0)
    else:
        chosen_cluster = int(chosen_cluster)
        st.write(f"Chosen cluster: {chosen_cluster}")

        initialize_widgets()

        current_cluster_df = st.session_state.chosen_cluster_df
        current_cluster_df = current_cluster_df[current_cluster_df["cluster"] == chosen_cluster].copy()
        current_cluster_df, cluster_samples = cluster_current_split(current_cluster_df)

        print(len(current_cluster_df), len(cluster_samples))
        construct_cluster_collages(cluster_samples)

        st.session_state.chosen_cluster_df = current_cluster_df

print("Initializing Style Quiz Streamlit Demo")
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

def initialize_widgets():
    radio_value = st.radio("Choose cluster:", ["0", "1", "Quit"], index=None, key="chosen_cluster", on_change=handle_radio_choice)

    image_widgets = []
    text_widgets = []
    for i in range(NUM_CLUSTERS):
        text_widgets.append(st.empty())
        image_widgets.append(st.empty())

    # Initialize the placements of the image and text widgets
    st.session_state.image_widgets = image_widgets
    st.session_state.text_widgets = text_widgets



if not st.session_state.initialized:
    st.session_state.initialized = True
    initialize_widgets()

    outfits_df = pd.read_pickle(OUTFIT_EMBEDDINGS_DF_PATH)
    outfits_to_lead_picture_id_dict = outfits_df.set_index("id")["lead_picture_id"].to_dict()
    st.session_state.outfits_to_lead_picture_id_dict = outfits_to_lead_picture_id_dict
    print("Loaded outfit embeddings dataframe")

    current_cluster_df, cluster_samples = cluster_current_split(outfits_df.dropna().copy())
    construct_cluster_collages(cluster_samples)
    st.session_state.chosen_cluster_df = current_cluster_df

    chosen_cluster = None
    print("Finished initializing")


