import random
import sys

from typing import List
import textwrap
from PIL import Image, ImageFont, ImageDraw
import matplotlib.pyplot as plt
import os

import traceback

def display_images(image_paths,
                   product_info: List[dict] = None,
                   columns=5,
                   width=20,
                   height=8,
                   max_images=15,
                   font_size=8,
                   text_wrap_length=35,
                   font_path=None,
                   plot_figure=True):
    if not image_paths:
        print("No images to display.")
        return
    if len(image_paths) > max_images:
        print(f"Showing {max_images} images of {len(image_paths)}:")
        image_paths = image_paths[0:max_images]
    height = max(height, len(image_paths) // columns * height)
    # height = height

    if plot_figure:
        fig = plt.figure(figsize=(width, height))
        # open images
        images = [Image.open(f) for f in image_paths]
        for i, image in enumerate(images):
            ax = plt.subplot(int(len(images) / columns + 1), columns, i + 1)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.box(False)
            ax.imshow(image)

            info = product_info[i] if product_info else {}
            display_text = ''
            for k,v in info.items():
                text = textwrap.wrap(str(v), text_wrap_length)
                display_text += '{}\n'.format("\n".join(text))
            ax.text(0, -0.1, display_text,
                        fontfamily='sans serif',
                        fontsize=font_size,
                        transform= ax.transAxes,
                        verticalalignment='top')
        fig.tight_layout(h_pad=3, w_pad=0.1)
        return fig
    
    RESIZE_SIZE = (255, 255)
    
    #print(f"Submitted number of images: {len(image_paths)}")
    #traceback.print_stack(file=sys.stdout)

    # Some of the images are not valid, so we need to skip them
    images = []
    for image_path in image_paths:
        try:
            images.append(Image.open(image_path))
        except:
            print(f"Error opening image at {image_path}")
    
    for img in images:
        img.thumbnail(RESIZE_SIZE)
    img_width, img_height = RESIZE_SIZE#images[0].size
    #rows = int(len(images) / columns + 1)
    rows = len(images) // columns + (len(images) % columns > 0)
    collage_width = columns * img_width
    collage_height = rows * img_height

    collage = Image.new('RGB', (collage_width, collage_height), color=(255, 255, 255))

    if font_path:
        font = ImageFont.truetype(font_path, font_size)
    else:
        font = ImageFont.load_default()

    for i, image in enumerate(images):
        row, col = divmod(i, columns)
        paste_width, paste_height = image.size
        x_offset = (img_width - paste_width) // 2
        x, y = col * img_width + x_offset, row * img_height
        collage.paste(image, (x, y))

        if product_info:
            info = product_info[i]
            display_text = ''
            for k, v in info.items():
                text = textwrap.wrap(str(v), text_wrap_length)
                display_text += '{}\n'.format("\n".join(text))

            draw = ImageDraw.Draw(collage)
            draw.multiline_text((x, y - 0.1 * img_height), display_text, font=font, fill=(0, 0, 0))

    return collage


def display_image_ids(image_ids, **kwargs):
    IMAGES_PATH = r"resources\data\dataset\images"
    display_image_paths = [os.path.join(IMAGES_PATH, f"{x}") for x in image_ids]
    return display_images(display_image_paths, columns=9, max_images=27, plot_figure=False, **kwargs)