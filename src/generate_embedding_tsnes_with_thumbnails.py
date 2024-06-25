import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import numpy as np

from PIL import Image, ImageDraw 
from tqdm.notebook import tqdm
import os

IMAGES_PATH = r"resources\data\dataset\images"
MAX_IMAGE_WIDTH, MAX_IMAGE_HEIGHT = 32, 256

def tsne_to_image_coordinates(tsne_x, tsne_y, img_width, img_height, x_min, x_max, y_min, y_max):
    img_x = ((tsne_x - x_min) / (x_max - x_min)) * img_width
    img_y = img_height - ((tsne_y - y_min) / (y_max - y_min)) * img_height
    return int(img_x), int(img_y)

def paste_thumbnail_on_image(image, thumbnail_path, x, y, max_width, max_height):
    thumbnail = Image.open(thumbnail_path)
    thumbnail.thumbnail((max_width, max_height))
    image.paste(thumbnail, (x, y))
    return image

def draw_thumbnails_on_scatter_plot(scatter_plot, tsne_df, FIGURE_SAVE_PATH="reports/figures/outfit_tsne.png"):
    # Use information from the scatter plot to map t-SNE coordinates to image coordinates
    image = Image.open(FIGURE_SAVE_PATH)
    image_width, image_height = image.size

    x_min, x_max = scatter_plot.get_xlim()
    y_min, y_max = scatter_plot.get_ylim()

    for row in tqdm(tsne_df.sample(100).itertuples()):
        image_path = os.path.join(IMAGES_PATH, row.lead_picture_id)
        img_x, img_y = tsne_to_image_coordinates(row.TSNE1, row.TSNE2, image_width, image_height, x_min, x_max, y_min, y_max)
        image = paste_thumbnail_on_image(image, image_path, img_x, img_y, MAX_IMAGE_WIDTH, MAX_IMAGE_HEIGHT)

    return image

def generate_tsne_diagram(outfits_df, embedding_column, show_plot=True, save_path="reports/figures/outfit_tsne.png"):
    embeddings = np.array(outfits_df[embedding_column].values.tolist())
    ids = outfits_df["id"].values

    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(embeddings)
    tsne_df = pd.DataFrame(tsne_results, columns=['TSNE1', 'TSNE2'])

    tsne_df["id"] = ids
    tsne_df = tsne_df.merge(outfits_df[["id", "category", "lead_picture_id"]], on="id").reset_index()
    tsne_df["category"] = tsne_df["category"].apply(lambda x: x[0] if len(x) > 0 else None)

    plt.figure(figsize=(16, 8))
    scatter_plot = sns.scatterplot(x='TSNE1', y='TSNE2', hue='category', data=tsne_df)
    plt.title(None)
    plt.xlabel(None)
    plt.ylabel(None)
    plt.legend(title='Category')
    sns.despine()
    plt.tight_layout()
    plt.axis('off')

    plt.savefig(save_path, bbox_inches='tight')

    if show_plot:
        plt.show()
    
    return tsne_df, scatter_plot
