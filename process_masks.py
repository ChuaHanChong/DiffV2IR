import argparse
import os

import numpy as np
from PIL import Image
from tqdm import tqdm


def process_masks(input_folder, output_folder):

    masks_file_list = os.listdir(input_folder)

    for mask_file in tqdm(masks_file_list, desc=f"Processing masks in {input_folder}"):
        mask_path = os.path.join(input_folder, mask_file)
        img_list = []
        img = np.array(Image.open(os.path.join(mask_path, "0.png")))
        img_list.append(img)
        image_name = mask_file + ".png"
        os.makedirs(output_folder, exist_ok=True)
        save_path = os.path.join(output_folder, image_name)
        new_img = np.zeros_like(img_list[0])
        Image.fromarray(new_img).save(save_path)
        image_pil = Image.open(save_path)
        image_pil = image_pil.convert("RGB")

        for _, _, files in os.walk(mask_path):
            for image in files:
                if image.endswith(".png"):
                    image_path = os.path.join(mask_path, image)
                    img = Image.open(image_path)
                    img_rgb = img.convert("RGB")
                    for i in range(img.size[0]):
                        for j in range(img.size[1]):
                            if img_rgb.getpixel((i, j)) == (255, 255, 255):
                                image_pil.putpixel((i, j), (255, 255, 255))

        image_pil.convert("RGB")
        image_pil.save(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_folder", type=str, required=True, help="Path to the input folder containing mask subfolders"
    )
    parser.add_argument(
        "--output_folder", type=str, required=True, help="Path to the output folder to save processed masks"
    )
    args = parser.parse_args()

    process_masks(args.input_folder, args.output_folder)
