
import os
import sys
sys.path.append(os.getcwd())
from src.display_images import display_image_ids


import streamlit as st
import time
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

OUTFIT_EMBEDDINGS_DF_PATH = r"resources\data\outfit_embeddings_triplets_df.pkl"
REPRESENTATION_COLUMN = "outfit_embeddings"
NUM_SAMPLES_PER_CLUSTER = 27

START_INDEX = 19580
NUM_TO_CONVERGENCE = 30

# Note: not implemented for more than 2 clusters yet
NUM_CLUSTERS = 2

def flatten(lst):
    return [item for el in lst for item in (flatten(el) if isinstance(el, list) else [el])]

def cluster_current_split(current_cluster_df, cluster_index_tree):
    cluster_samples = []
    for i in range(NUM_CLUSTERS):
        cluster_indexes = cluster_index_tree[i]

        # Linkage assigns temporary indexes to the parent clusters, so we need to filter out the indexes that are out of bounds
        # The indexes below the length of the dataframe are the actual outfit indexes though
        outfit_indexes = [index for index in flatten(cluster_indexes) if index < len(current_cluster_df)]
        cluster_outfits = current_cluster_df.iloc[outfit_indexes].copy()
        cluster_embeddings = np.stack(cluster_outfits[REPRESENTATION_COLUMN].values)
        num_cluster_samples = min(NUM_SAMPLES_PER_CLUSTER, len(cluster_outfits))
        
        # Properly represent the diversity of the cluster by applying KMeans to the embeddings
        cluster_kmedoids = KMedoids(n_clusters=num_cluster_samples, random_state=1, init="k-medoids++").fit(cluster_embeddings)
        cluster_representation = cluster_outfits.iloc[cluster_kmedoids.medoid_indices_].reset_index()
        cluster_samples.append(cluster_representation)

        st.session_state.text_widgets[i].write(f"Cluster {i}: {len(cluster_outfits)} outfits")

    return current_cluster_df, cluster_index_tree, cluster_samples

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
        current_cluster_hierarchy = st.session_state.current_cluster_hierarchy
        chosen_cluster_list = current_cluster_hierarchy[chosen_cluster]
        
        if len(flatten(chosen_cluster_list)) <= NUM_TO_CONVERGENCE:
            st.session_state.termination_widget.write(f"Reached the end of the cluster hierarchy with cluster {chosen_cluster}!")
            raise SystemExit(0)
        
        current_cluster_df, current_cluster_hierarchy, cluster_samples = cluster_current_split(current_cluster_df, chosen_cluster_list)

        construct_cluster_collages(cluster_samples)

        st.session_state.chosen_cluster_df = current_cluster_df
        st.session_state.current_cluster_hierarchy = current_cluster_hierarchy

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
    st.session_state.termination_widget = st.empty()

def compute_outfit_clusters(outfits_df):
    input_embeddings = np.stack(outfits_df[REPRESENTATION_COLUMN].values)
    Z = linkage(input_embeddings, 'ward', metric='euclidean')

    number_of_embeddings = len(input_embeddings)
    cluster_hierarchy = {i : [i] for i in range(number_of_embeddings)}

    for index, (index_0, index_1, distance, number_in_cluster) in enumerate(Z):
        if index_0 in cluster_hierarchy and index_1 in cluster_hierarchy:
            new_cluster_index = number_of_embeddings + index
            cluster_hierarchy[new_cluster_index] = [cluster_hierarchy[index_0], cluster_hierarchy[index_1]]

    return cluster_hierarchy[START_INDEX]


if not st.session_state.initialized:
    st.session_state.initialized = True
    initialize_widgets()

    outfits_df = pd.read_pickle(OUTFIT_EMBEDDINGS_DF_PATH)
    outfits_to_lead_picture_id_dict = outfits_df.set_index("id")["lead_picture_id"].to_dict()
    current_cluster_hierarchy = compute_outfit_clusters(outfits_df)
    print(len(current_cluster_hierarchy))

    st.session_state.outfits_to_lead_picture_id_dict = outfits_to_lead_picture_id_dict
    print("Loaded outfit embeddings dataframe")

    current_cluster_df, current_cluster_hierarchy, cluster_samples = cluster_current_split(outfits_df.dropna().copy(), current_cluster_hierarchy)
    construct_cluster_collages(cluster_samples)
    st.session_state.chosen_cluster_df = current_cluster_df
    st.session_state.current_cluster_hierarchy = current_cluster_hierarchy

    chosen_cluster = None
    print("Finished initializing")


