import argparse
import os
from typing import Any, Dict, List

import torch

import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


def write_masks_to_folder(masks: List[Dict[str, Any]], path: str) -> None:
    header = "id,area,bbox_x0,bbox_y0,bbox_w,bbox_h,point_input_x,point_input_y,predicted_iou,stability_score,crop_box_x0,crop_box_y0,crop_box_w,crop_box_h"  # noqa
    metadata = [header]
    for i, mask_data in enumerate(masks):
        mask = mask_data["segmentation"]
        filename = f"{i}.png"
        Image.fromarray(mask).save(os.path.join(path, filename))
        mask_metadata = [
            str(i),
            str(mask_data["area"]),
            *[str(x) for x in mask_data["bbox"]],
            *[str(x) for x in mask_data["point_coords"][0]],
            str(mask_data["predicted_iou"]),
            str(mask_data["stability_score"]),
            *[str(x) for x in mask_data["crop_box"]],
        ]
        row = ",".join(mask_metadata)
        metadata.append(row)
    metadata_path = os.path.join(path, "metadata.csv")
    with open(metadata_path, "w") as f:
        f.write("\n".join(metadata))

    return


def predict(args):
    device = torch.device("cuda:2")

    sam2_checkpoint = "/data/hanchong/DiffV2IR/sam2/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)

    mask_generator = SAM2AutomaticMaskGenerator(sam2)

    df = pd.read_csv(args.df)

    with torch.inference_mode():
        for input_path, output_path in tqdm(df.itertuples(index=False), total=len(df)):
            try:
                input_path = Path(input_path)
                output_path = Path(output_path)
                output_path.mkdir(parents=True, exist_ok=True)

                image = np.array(Image.open(input_path).convert("RGB"))
                masks = mask_generator.generate(image)

                write_masks_to_folder(masks, str(output_path))

            except Exception as e:
                print(f"Error processing image {str(input_path)}: {e}")
                continue


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--df", type=str, required=True)
    args = argparser.parse_args()
    predict(args)
