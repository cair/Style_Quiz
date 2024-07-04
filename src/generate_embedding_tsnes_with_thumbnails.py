import pandas as pd
import matplotlib.pyplot as plt
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

def draw_points_on_scatter_plot(scatter_plot, points, color='black', size=50, marker='X'):
    for point in points:
        scatter_plot.scatter(point[0], point[1], s=size, c=color, marker=marker)
    return scatter_plot

def draw_thumbnails_on_scatter_plot(scatter_plot, tsne_df, num_thumbnails=-1, FIGURE_SAVE_PATH="reports/figures/outfit_tsne.png"):
    # Use information from the scatter plot to map t-SNE coordinates to image coordinates
    image = Image.open(FIGURE_SAVE_PATH)
    image_width, image_height = image.size

    x_min, x_max = scatter_plot.get_xlim()
    y_min, y_max = scatter_plot.get_ylim()

    num_thumbnails = num_thumbnails if num_thumbnails > 0 else tsne_df.dropna(subset=["lead_picture_id"]).shape[0]
    # Not all outfits have a lead picture, so we need to drop the ones that do not have it
    for row in tqdm(tsne_df.dropna(subset=["lead_picture_id"]).sample(num_thumbnails).itertuples()):
        image_path = os.path.join(IMAGES_PATH, row.lead_picture_id)
        img_x, img_y = tsne_to_image_coordinates(row.TSNE1, row.TSNE2, image_width, image_height, x_min, x_max, y_min, y_max)
        try:
            image = paste_thumbnail_on_image(image, image_path, img_x, img_y, MAX_IMAGE_WIDTH, MAX_IMAGE_HEIGHT)
        except Exception as e:
            print(f"Error while pasting thumbnail {image_path} on image: {e}")

    return image

from sklearn.manifold import TSNE

def generate_tsne_diagram(outfits_df, embedding_column, mark_points=None, return_tsne=False, show_plot=True, hue_column="category", save_path="reports/figures/outfit_tsne.png"):
    embeddings = np.array(outfits_df[embedding_column].values.tolist())
    ids = outfits_df["id"].values

    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(embeddings)
    tsne_df = pd.DataFrame(tsne_results, columns=['TSNE1', 'TSNE2'])

    tsne_df["id"] = ids
    tsne_df = tsne_df.merge(outfits_df[["id", hue_column, "lead_picture_id"]], on="id").reset_index()
    #tsne_df["category"] = tsne_df["category"].apply(lambda x: x[0] if len(x) > 0 else None)
    # If hue column is "category", we need to extract the first element of the list
    if type(tsne_df[hue_column].iloc[0]) == list:
        tsne_df[hue_column] = tsne_df[hue_column].apply(lambda x: x[0] if len(x) > 0 else None)



    plt.figure(figsize=(16, 8))
    #scatter_plot = sns.scatterplot(x='TSNE1', y='TSNE2', hue="category", data=tsne_df)
    if OTHER_VALUE in tsne_df[hue_column].unique():
        other_points = tsne_df[tsne_df[hue_column] == OTHER_VALUE]
        scatter_plot = sns.scatterplot(x='TSNE1', y='TSNE2', hue=hue_column, data=other_points)
        category_points = tsne_df[tsne_df[hue_column] != OTHER_VALUE]
        sns.scatterplot(x='TSNE1', y='TSNE2', hue=hue_column, data=category_points)
    else:
        scatter_plot = sns.scatterplot(x='TSNE1', y='TSNE2', hue=hue_column, data=tsne_df)


    # Meant to mark points on the scatter plot that are of interest, initially used to mark the outfits that don't have any positive examples from triplet loss
    if mark_points is not None:
        tsne_df["marked"] = mark_points
        marked_points = tsne_df[tsne_df["marked"] == True]
        sns.scatterplot(x='TSNE1', y='TSNE2', hue='category', data=marked_points, s=50, marker='X', edgecolor='black', linewidth=1)

    plt.title(None)
    plt.xlabel(None)
    plt.ylabel(None)
    plt.legend(title=hue_column.capitalize())
    sns.despine()
    plt.tight_layout()
    plt.axis('off')

    plt.savefig(save_path, bbox_inches='tight')

    if show_plot:
        plt.show()
    
    if return_tsne:
        return tsne_df, scatter_plot, tsne

    return tsne_df, scatter_plot


from sklearn.decomposition import PCA

def generate_pca_diagram(outfits_df, embedding_column, extra_embeddings=None, mark_points=None, return_pca=False, show_plot=True, hue_column="category", save_path="reports/figures/outfit_pca.png"):
    embeddings = np.array(outfits_df[embedding_column].values.tolist())
    ids = outfits_df["id"].values
    pca = PCA(n_components=2, random_state=42)
    pca_results = pca.fit_transform(embeddings)
    pca_df = pd.DataFrame(pca_results, columns=['PC1', 'PC2'])
    pca_df["id"] = ids
    pca_df = pca_df.merge(outfits_df[["id", hue_column, "lead_picture_id"]], on="id").reset_index()
    pca_df["category"] = pca_df["category"].apply(lambda x: x[0] if len(x) > 0 else None)
    
    plt.figure(figsize=(16, 8))
    scatter_plot = sns.scatterplot(x='PC1', y='PC2', hue="category", data=pca_df)
    
    if mark_points is not None:
        pca_df["marked"] = mark_points
        marked_points = pca_df[pca_df["marked"] == True]
        sns.scatterplot(x='PC1', y='PC2', hue='category', data=marked_points, s=50, marker='X', edgecolor='black', linewidth=1)
    
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
   
    if return_pca:
        return pca_df, scatter_plot, pca
    return pca_df, scatter_plot

from sklearn.decomposition import TruncatedSVD

OTHER_VALUE = "other"

def generate_svd_diagram(outfits_df, embedding_column, extra_embeddings=None, mark_points=None, return_svd=False, show_plot=True, hue_column="category", save_path="reports/figures/outfit_svd.png"):
    embeddings = np.array(outfits_df[embedding_column].values.tolist())
    ids = outfits_df["id"].values
    svd = TruncatedSVD(n_components=2, random_state=42)
    svd_results = svd.fit_transform(embeddings)
    svd_df = pd.DataFrame(svd_results, columns=['SV1', 'SV2'])
    svd_df["id"] = ids
    svd_df = svd_df.merge(outfits_df[["id", hue_column, "lead_picture_id"]], on="id").reset_index()

    # If hue column is "category", we need to extract the first element of the list
    if type(svd_df[hue_column].iloc[0]) == list:
        svd_df[hue_column] = svd_df[hue_column].apply(lambda x: x[0] if len(x) > 0 else None)
    
    plt.figure(figsize=(16, 8))
    if OTHER_VALUE in svd_df[hue_column].unique():
        other_points = svd_df[svd_df[hue_column] == OTHER_VALUE]
        scatter_plot = sns.scatterplot(x='SV1', y='SV2', hue=hue_column, data=other_points)
        category_points = svd_df[svd_df[hue_column] != OTHER_VALUE]
        sns.scatterplot(x='SV1', y='SV2', hue=hue_column, data=category_points)
    else:
        scatter_plot = sns.scatterplot(x='SV1', y='SV2', hue=hue_column, data=svd_df)

    if mark_points is not None:
        svd_df["marked"] = mark_points
        marked_points = svd_df[svd_df["marked"] == True]
        sns.scatterplot(x='SV1', y='SV2', hue=hue_column, data=marked_points, s=50, marker='X', edgecolor='black', linewidth=1)
    if extra_embeddings is not None:
        svd_embeddings = svd.transform(extra_embeddings)
        plt.scatter(svd_embeddings[:, 0], svd_embeddings[:, 1], c='black', marker='X', s=50)
    
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
   
    if return_svd:
        return svd_df, scatter_plot, svd
    return svd_df, scatter_plot