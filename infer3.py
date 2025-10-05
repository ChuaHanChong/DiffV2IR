from __future__ import annotations

import json
import math
import os
import random
import shutil
import sys
from argparse import ArgumentParser

import einops
import k_diffusion as K
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image, ImageOps
from torch import autocast
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm


sys.path.append("./stable_diffusion")
from stable_diffusion.ldm.util import instantiate_from_config


class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, z, sigma, cond, uncond, text_cfg_scale, image_cfg_scale, seg_cfg_scale):
        cfg_z = einops.repeat(z, "1 ... -> n ...", n=4)
        cfg_sigma = einops.repeat(sigma, "1 ... -> n ...", n=4)
        cfg_cond = {
            "c_crossattn": [torch.cat([cond["c_crossattn"][0], uncond["c_crossattn"][0], uncond["c_crossattn"][0], uncond["c_crossattn"][0]])],
            "c_concat1": [torch.cat([cond["c_concat1"][0], cond["c_concat1"][0], uncond["c_concat1"][0], uncond["c_concat1"][0]])],
            "c_concat2": [torch.cat([cond["c_concat2"][0], cond["c_concat2"][0], cond["c_concat2"][0], uncond["c_concat2"][0]])],
        }
        out_cond, out_img_cond, out_seg_cond, out_uncond = self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond).chunk(4)
        if seg_cfg_scale == 0:
            return out_uncond + text_cfg_scale * (out_cond - out_img_cond) + image_cfg_scale * (out_img_cond - out_uncond)
        return out_uncond + text_cfg_scale * (out_cond - out_img_cond) + image_cfg_scale * (out_img_cond - out_seg_cond) + seg_cfg_scale * (out_seg_cond - out_uncond)


def get_text_for_image(image_filename, json_file):
    with open(json_file, "r", encoding="utf-8") as infile:
        image_text_data = json.load(infile)

    if image_filename in image_text_data:
        return image_text_data[image_filename]
    else:
        return None


def load_model_from_config(config, ckpt, vae_ckpt=None, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    if vae_ckpt is not None:
        print(f"Loading VAE from {vae_ckpt}")
        vae_sd = torch.load(vae_ckpt, map_location="cpu")["state_dict"]
        sd = {k: vae_sd[k[len("first_stage_model.") :]] if k.startswith("first_stage_model.") else v for k, v in sd.items()}
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    return model


def load_demo_image(image_size, device, img_url):

    raw_image = Image.open(img_url).convert("RGB")

    w, h = raw_image.size

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]
    )
    image = transform(raw_image).unsqueeze(0)
    return image

category_mapping = {
    "C00": "Barge",
    "C02": "Bulk Carrier",
    "C04": "Cargo Ship",
    "C05": "Container Ship",
    "C07": "Cruise",
    "C08": "Dredger",
    "C09": "Ferry",
    "C10": "Fishing Vessel",
    "C11": "Law Enforcement",
    "C12": "Military Vessel",
    "C13": "Pilot Vessel",
    "C14": "RORO",
    "C15": "Sailing Vessel",
    "C17": "Supply Vessel",
    "C18": "Tanker",
    "C19": "Tugboat",
    "C20": "Yacht",
}

def main():
    parser = ArgumentParser()
    parser.add_argument("--resolution", default=512, type=int)
    parser.add_argument("--steps", default=100, type=int)
    parser.add_argument("--config", default="configs/generate.yaml", type=str)
    parser.add_argument("--ckpt", default="", type=str)
    parser.add_argument("--vae-ckpt", default=None, type=str)
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--cfg-text", default=7.5, type=float)
    parser.add_argument("--cfg-image", default=1.5, type=float)
    parser.add_argument("--cfg-seg", default=1.5, type=float)
    parser.add_argument("--no-seg", action="store_true", help="Disable segmentation conditioning")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    # os.makedirs('/home/jovyan/.cache/torch/hub/checkpoints/')
    # shutil.copy("checkpoint_liberty_with_aug.pth","/home/jovyan/.cache/torch/hub/checkpoints/")

    os.makedirs(args.output, exist_ok=True)
    config = OmegaConf.load(args.config)
    model = load_model_from_config(config, args.ckpt, args.vae_ckpt)
    model.eval().cuda()
    model_wrap = K.external.CompVisDenoiser(model)
    model_wrap_cfg = CFGDenoiser(model_wrap)
    null_token = model.get_learned_conditioning([""])
    seed = random.randint(0, 100000) if args.seed is None else args.seed
    for root, _, files in os.walk(args.input):
        caption = "turn the visible image of " + category_mapping[args.input.split("/")[-1]] + " into infrared"
        for file in tqdm(files, desc=f"Processing files in {root}"):
            if os.path.exists(os.path.join(args.output, file)):
                continue
            input_image = Image.open(os.path.join(args.input, file)).convert("RGB")
            if args.no_seg:
                input_seg = input_image.copy()
            else:
                input_seg = Image.open(os.path.join(args.input + "_seg", file.split(".")[0] + ".png")).convert("RGB")
            width, height = input_image.size
            factor = args.resolution / max(width, height)
            factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
            width = int((width * factor) // 64) * 64
            height = int((height * factor) // 64) * 64
            input_image = ImageOps.fit(input_image, (width, height), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
            input_seg = ImageOps.fit(input_seg, (width, height), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))

            with torch.no_grad(), autocast("cuda"), model.ema_scope():
                cond = {}
                cond["c_crossattn"] = [model.get_learned_conditioning([caption])]
                input_image = 2 * torch.tensor(np.array(input_image)).float() / 255 - 1
                input_seg = 2 * torch.tensor(np.array(input_seg)).float() / 255 - 1
                input_image = rearrange(input_image, "h w c -> 1 c h w").to(model.device)
                input_seg = rearrange(input_seg, "h w c -> 1 c h w").to(model.device)
                cond["c_concat1"] = [model.encode_first_stage(input_image).mode()]
                cond["c_concat2"] = [model.encode_first_stage(input_seg).mode()]

                uncond = {}
                uncond["c_crossattn"] = [null_token]
                uncond["c_concat1"] = [torch.zeros_like(cond["c_concat1"][0])]
                uncond["c_concat2"] = [torch.zeros_like(cond["c_concat2"][0])]

                sigmas = model_wrap.get_sigmas(args.steps)

                extra_args = {
                    "cond": cond,
                    "uncond": uncond,
                    "text_cfg_scale": args.cfg_text,
                    "image_cfg_scale": args.cfg_image,
                    "seg_cfg_scale": args.cfg_seg,
                }
                torch.manual_seed(seed)
                z = torch.randn_like(cond["c_concat1"][0]) * sigmas[0]
                z = K.sampling.sample_euler_ancestral(model_wrap_cfg, z, sigmas, extra_args=extra_args)
                x = model.decode_first_stage(z)
                x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
                x = 255.0 * rearrange(x, "1 c h w -> h w c")
                edited_image = Image.fromarray(x.type(torch.uint8).cpu().numpy())
            edited_image.save(os.path.join(args.output, file))


if __name__ == "__main__":
    main()
