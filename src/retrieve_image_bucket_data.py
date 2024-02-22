# Migrated from FREja

from google.cloud import storage
import torchvision.io as io
import os
from PIL import Image
from resources.constants import *

import asyncio
from tqdm import tqdm

async def get_file_size_async(picture_id, bucket):
    blob = await bucket.get_blob(f"pictures/{picture_id}")
    return blob.size

#Get file sizes of picture ids, allows for detection of corrupted images
async def get_file_sizes(picture_ids):
    bucket = get_bucket()

    tasks = []
    for picture_id in tqdm(picture_ids):
        task = asyncio.ensure_future(get_file_size_async(picture_id, bucket))
        tasks.append(task)

    #sizes = await asyncio.gather(*tasks)
    sizes = []
    with tqdm(total=len(picture_ids)) as pbar:
        for coroutine in asyncio.as_completed(tasks):
            size = await coroutine
            sizes.append(size)
            pbar.update(1)
    return sizes

def get_file_sizes_asyn(picture_ids):
    loop = asyncio.get_event_loop()
    sizes = loop.run_until_complete(get_file_sizes(picture_ids))
    return sizes

def retrieve_picture_ids():
    storage_client = storage.Client(project='personal-sandbox-1333')
    bucket = storage_client.get_bucket("fjong")
    blobs_list = [blob.name.split("/")[-1] for blob in bucket.list_blobs()]
    return blobs_list


def get_bucket():
    storage_client = storage.Client(project='personal-sandbox-1333')
    bucket = storage_client.get_bucket("fjong")
    return bucket


def download_picture(bucket, picture_id, save_file_name="Temp.jpg", image_format="torch"):
    blob = bucket.get_blob(f"pictures/{picture_id}")
    blob.download_to_filename(save_file_name)
    try:
        if image_format == "torch":
            return io.read_image(save_file_name, mode=io.ImageReadMode.RGB)
        elif image_format == "PIL":
            return Image.open(save_file_name)
        elif image_format == "None":
            return None
    except:
        print(f"Error with image_id: {picture_id}, using default image... FIX THIS LATER")
        default_image = Image.new("RGB", (1448, 2048), (255, 255, 255))
        default_image.save(PLACEHOLDER_IMAGE_NAME)
        if image_format == "torch":
            default_image_io = io.read_image(PLACEHOLDER_IMAGE_NAME, mode=io.ImageReadMode.RGB)
            os.remove(PLACEHOLDER_IMAGE_NAME)
            return default_image_io
        elif image_format == "PIL":
            return default_image

def download_to_folder(picture_ids, folder_name):
    bucket = get_bucket()
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    for picture_id in tqdm(picture_ids):
        save_file_name = f"{folder_name}/{picture_id}"
        download_picture(bucket, picture_id, save_file_name, image_format="None")

import shutil

def download_temp_batch(picture_ids, temp_folder_name):
    bucket = get_bucket()
    if not os.path.exists(temp_folder_name):
        os.makedirs(temp_folder_name)
    
    batch_pictures = []
    for picture_id in tqdm(picture_ids):
        save_file_name = f"{temp_folder_name}/{picture_id}"
        batch_pictures.append(download_picture(bucket, picture_id, save_file_name, image_format="torch"))
    
    shutil.rmtree(temp_folder_name)
    return batch_pictures

def return_default_image(image_format):
    default_image = Image.new("RGB", (1448, 2048), (255, 255, 255))
    default_image.save(PLACEHOLDER_IMAGE_NAME)
    if image_format == "torch":
        default_image_io = io.read_image(PLACEHOLDER_IMAGE_NAME, mode=io.ImageReadMode.RGB)
        os.remove(PLACEHOLDER_IMAGE_NAME)
        return default_image_io
    elif image_format == "PIL":
        return default_image

# Testing async download of images below
# import pandas as pd

# if __name__ == '__main__':
#     pictures_df = pd.read_pickle(FILTERED_PICTURES_NAME)
#     picture_ids = pictures_df["id"].tolist()
#     sizes = asyncio.run(get_file_sizes(picture_ids))