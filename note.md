# Note

## Development

```bash
conda create -n DiffV2IR python=3.12
conda activate DiffV2IR
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
cd segment-anything-2 && pip install -e . && python setup.py build_ext --inplace && cd ..
pip install einops 
pip install git+https://github.com/crowsonkb/k-diffusion.git 
pip install timm 
pip install fairscale 
pip install transformers 
pip install pytorch_lightning 
pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
pip install omegaconf
pip install pandas
```

```bash
% tmux a -t 8 9 10

python amg.py --df /data/hanchong/maritime-vessel-dataset-infrared/In-distribution_Auto-Masks/val_batch_0.csv

python process_mask

CUDA_VISIBLE_DEVICES=2 python infer.py --ckpt /data/hanchong/DiffV2IR/IR-500K/IR-500k/finetuned_checkpoints/FLIR.ckpt --input /data/hanchong/maritime-vessel-dataset-infrared/In-distribution/val/C00 --output /data/hanchong/maritime-vessel-dataset-infrared/In-distribution_IR-FLIR/val/C00

CUDA_VISIBLE_DEVICES=1 python infer.py --ckpt /data/hanchong/DiffV2IR/IR-500K/IR-500k/finetuned_checkpoints/M3FD.ckpt --input /data/hanchong/maritime-vessel-dataset-infrared/In-distribution/val/C00 --output /data/hanchong/maritime-vessel-dataset-infrared/In-distribution_IR-M3FD/val/C00
```

## References

https://arxiv.org/pdf/2503.19012
https://github.com/LidongWang-26/DiffV2IR
https://github.com/LidongWang-26/DiffV2IR/issues/13
https://github.com/facebookresearch/segment-anything/blob/main/scripts/amg.py
https://github.com/facebookresearch/sam2/blob/main/notebooks/automatic_mask_generator_example.ipynb
