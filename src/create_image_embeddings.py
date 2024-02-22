# Note: this file once used to save embeddings to cloud storage, see implementation in FREja_APIs

import os
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import torch
from torchvision.io import read_image
import torchvision

from src.retrieve_image_bucket_data import retrieve_picture_ids, get_bucket, download_picture, return_default_image
from resources.constants import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_image_series_torchvision(image_series, bucket):
    images = []
    local_pictures = os.listdir(DATASET_IMAGES_FOLDER)
    for image_id in image_series:
        image_path = os.path.join(DATASET_IMAGES_FOLDER, image_id)
        if image_id not in local_pictures:
            download_picture(bucket, image_id, save_file_name=image_path)
        try:
            torch_image = read_image(image_path, mode=torchvision.io.ImageReadMode.RGB)
        except:
            print(f"Error when reading image {image_id}, returning default image")
            torch_image = return_default_image(image_format="torch")
        images.append(torch_image)
    return images

class Embedding_Config():
    class exception_hack(Exception):
        pass

    
    class Parasite_Module(torch.nn.Module):
        def __init__(self, host_layer, include_host=True):
            super().__init__()
            self.host_layer = host_layer
            self.include_host = include_host

        def forward(self, x):
            if self.include_host:
                x = self.host_layer(x)
            raise Embedding_Config.exception_hack(x)

    def __init__(self):
        self.use_cuda = None
        
        self.weights = None
        self.transforms = None
        self.model = None
        self.model_save_name = None
        self.model_save_folder = None
        self.bucket = get_bucket()

    def load_model(self):
        raise NotImplementedError

    def prepare_model(self):
        model = self.load_model()
        if self.use_cuda:
            model.to(DEVICE)
        model.eval()
        return model

    def print_config_summary(self):
        print(f"PKL name: {self.model_save_name} | Model: {type(self.model).__name__} | Weights: {type(self.weights).__name__} | {self.weights.name}")

    def load_images(self, image_series):
        image_tensors = load_image_series_torchvision(image_series, self.bucket)
        formatted_images = [self.transforms(test_img).unsqueeze(0) for test_img in image_tensors]
        formatted_images = torch.vstack(formatted_images)
        if self.use_cuda:
            formatted_images = formatted_images.to(DEVICE)
        return formatted_images
    
    def get_embeddings(self, image_tensors):
        with torch.no_grad():
            try:
                embeddings = self.model(image_tensors)
            except self.exception_hack as e:
                embeddings = e.args[0]
        # Reduce to half precision to save space and load time
        forward_embeddings = embeddings.cpu()#embeddings.half().cpu()
        return forward_embeddings.half()
    
    def save_embeddings(self, embeddings, picture_ids, save_path):
        save_path = os.path.join(save_path, self.model_save_folder)
        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        for embedding, picture_id in zip(embeddings, picture_ids):
            np.save(os.path.join(save_path, picture_id), embedding)


class EfficientNet_V2_L_final(Embedding_Config):
    def __init__(self):
        super().__init__()
        self.use_cuda = True

        self.weights = torchvision.models.EfficientNet_V2_L_Weights.IMAGENET1K_V1
        self.transforms = self.weights.transforms(antialias=True)
        self.model = self.prepare_model()
        self.model_save_name = f"{self.__class__.__name__}.pkl"
        self.model_save_folder = f"{self.__class__.__name__}"
        self.print_config_summary()
        
    def load_model(self):
        model = torchvision.models.efficientnet_v2_l(weights=self.weights)
        model.classifier[1] = self.Parasite_Module(model.classifier[1], include_host=False)
        return model

MAX_IMAGES_PER_BATCH = 256
def get_df_image_embeddings(base_df, data_path, config_class : Embedding_Config):
    if not os.path.isdir(DATASET_IMAGE_EMBEDDINGS_FOLDER):
        os.mkdir(DATASET_IMAGE_EMBEDDINGS_FOLDER)

    model_config = config_class()
    print(f"Generating embeddings from {type(model_config).__name__} to {model_config.model_save_name}...")
    
    number_of_df_splits = len(base_df) // MAX_IMAGES_PER_BATCH + 1
    split_dfs = np.array_split(base_df, number_of_df_splits)
    for df_split in tqdm(split_dfs):
        split_ids = df_split["picture.id"]
        df_images = model_config.load_images(split_ids)

        df_embeddings = model_config.get_embeddings(df_images).numpy()
        model_config.save_embeddings(df_embeddings, split_ids.values, DATASET_IMAGE_EMBEDDINGS_FOLDER)
        df_embeddings = list(df_embeddings)

        df_split["embeddings"] = df_embeddings
    embedding_df = pd.concat(split_dfs)
    embedding_df.to_pickle(os.path.join(data_path, model_config.model_save_name))
    return embedding_df