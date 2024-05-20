import os
import random
from PIL import Image

def create_image_collage(folder_path, output_path, num_images=1000, collage_width=2000, collage_height=1000):
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]
    selected_images = random.sample(image_files, min(num_images, len(image_files)))

    images = [Image.open(img) for img in selected_images]
    
    aspect_ratio = collage_width / collage_height
    num_images_width = int((num_images * aspect_ratio) ** 0.5)
    num_images_height = int(num_images ** 0.5 / aspect_ratio)

    if num_images_width * num_images_height < num_images:
        num_images_width += 1

    thumb_width = collage_width // num_images_width
    thumb_height = collage_height // num_images_height
    print(f'Grid size: {num_images_width}x{num_images_height}, Thumbnail size: {thumb_width}x{thumb_height}')
    collage_width = num_images_width * thumb_width
    collage_height = num_images_height * thumb_height
    
    collage_image = Image.new('RGB', (collage_width, collage_height))
    for index, img in enumerate(images):
        if index >= num_images_width * num_images_height:
            break
        img = img.resize((thumb_width, thumb_height), Image.Resampling.LANCZOS)
        x = (index % num_images_width) * thumb_width
        y = (index // num_images_width) * thumb_height
        collage_image.paste(img, (x, y))
    
    collage_image.save(output_path)
    collage_image.show()

folder_path = r'..\..\resources\data\dataset\images'
output_path = r'..\..\reports\figures\collage.jpg'
create_image_collage(folder_path, output_path)