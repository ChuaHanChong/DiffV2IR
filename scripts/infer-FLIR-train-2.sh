#CUDA_VISIBLE_DEVICES=0 python infer3.py \
#--ckpt /data/hanchong/DiffV2IR/IR-500K/IR-500k/finetuned_checkpoints/FLIR.ckpt \
#--input /data/hanchong/images-splitted-3/In-distribution/train/C00 \
#--output /data/hanchong/maritime-vessel-dataset-infrared/In-distribution_IR-FLIR-2/train/C00 \
#--steps=50 \
#--no-seg \
#--cfg-seg 0
#
#CUDA_VISIBLE_DEVICES=1 python infer3.py \
#--ckpt /data/hanchong/DiffV2IR/IR-500K/IR-500k/finetuned_checkpoints/FLIR.ckpt \
#--input /data/hanchong/images-splitted-3/In-distribution/train/C02 \
#--output /data/hanchong/maritime-vessel-dataset-infrared/In-distribution_IR-FLIR-2/train/C02 \
#--steps=50 \
#--no-seg \
#--cfg-seg 0
#
#CUDA_VISIBLE_DEVICES=2 python infer3.py \
#--ckpt /data/hanchong/DiffV2IR/IR-500K/IR-500k/finetuned_checkpoints/FLIR.ckpt \
#--input /data/hanchong/images-splitted-3/In-distribution/train/C04 \
#--output /data/hanchong/maritime-vessel-dataset-infrared/In-distribution_IR-FLIR-2/train/C04 \
#--steps=50 \
#--no-seg \
#--cfg-seg 0
#
#CUDA_VISIBLE_DEVICES=3 python infer3.py \
#--ckpt /data/hanchong/DiffV2IR/IR-500K/IR-500k/finetuned_checkpoints/FLIR.ckpt \
#--input /data/hanchong/images-splitted-3/In-distribution/train/C05 \
#--output /data/hanchong/maritime-vessel-dataset-infrared/In-distribution_IR-FLIR-2/train/C05 \
#--steps=50 \
#--no-seg \
#--cfg-seg 0
#
#CUDA_VISIBLE_DEVICES=2 python infer3.py \
#--ckpt /data/hanchong/DiffV2IR/IR-500K/IR-500k/finetuned_checkpoints/FLIR.ckpt \
#--input /data/hanchong/images-splitted-3/In-distribution/train/C07 \
#--output /data/hanchong/maritime-vessel-dataset-infrared/In-distribution_IR-FLIR-2/train/C07 \
#--steps=50 \
#--no-seg \
#--cfg-seg 0
#
#CUDA_VISIBLE_DEVICES=0 python infer3.py \
#--ckpt /data/hanchong/DiffV2IR/IR-500K/IR-500k/finetuned_checkpoints/FLIR.ckpt \
#--input /data/hanchong/images-splitted-3/In-distribution/train/C08 \
#--output /data/hanchong/maritime-vessel-dataset-infrared/In-distribution_IR-FLIR-2/train/C08 \
#--steps=50 \
#--no-seg \
#--cfg-seg 0
#
#CUDA_VISIBLE_DEVICES=1 python infer3.py \
#--ckpt /data/hanchong/DiffV2IR/IR-500K/IR-500k/finetuned_checkpoints/FLIR.ckpt \
#--input /data/hanchong/images-splitted-3/In-distribution/train/C09 \
#--output /data/hanchong/maritime-vessel-dataset-infrared/In-distribution_IR-FLIR-2/train/C09 \
#--steps=50 \
#--no-seg \
#--cfg-seg 0
#
#CUDA_VISIBLE_DEVICES=2 python infer3.py \
#--ckpt /data/hanchong/DiffV2IR/IR-500K/IR-500k/finetuned_checkpoints/FLIR.ckpt \
#--input /data/hanchong/images-splitted-3/In-distribution/train/C10 \
#--output /data/hanchong/maritime-vessel-dataset-infrared/In-distribution_IR-FLIR-2/train/C10 \
#--steps=50 \
#--no-seg \
#--cfg-seg 0
#
#CUDA_VISIBLE_DEVICES=3 python infer3.py \
#--ckpt /data/hanchong/DiffV2IR/IR-500K/IR-500k/finetuned_checkpoints/FLIR.ckpt \
#--input /data/hanchong/images-splitted-3/In-distribution/train/C11 \
#--output /data/hanchong/maritime-vessel-dataset-infrared/In-distribution_IR-FLIR-2/train/C11 \
#--steps=50 \
#--no-seg \
#--cfg-seg 0

#CUDA_VISIBLE_DEVICES=2 python infer3.py \
#--ckpt /data/hanchong/DiffV2IR/IR-500K/IR-500k/finetuned_checkpoints/FLIR.ckpt \
#--input /data/hanchong/images-splitted-3/In-distribution/train/C12 \
#--output /data/hanchong/maritime-vessel-dataset-infrared/In-distribution_IR-FLIR-2/train/C12 \
#--steps=50 \
#--no-seg \
#--cfg-seg 0

#CUDA_VISIBLE_DEVICES=2 python infer3.py \
#--ckpt /data/hanchong/DiffV2IR/IR-500K/IR-500k/finetuned_checkpoints/FLIR.ckpt \
#--input /data/hanchong/images-splitted-3/In-distribution/train/C13 \
#--output /data/hanchong/maritime-vessel-dataset-infrared/In-distribution_IR-FLIR-2/train/C13 \
#--steps=50 \
#--no-seg \
#--cfg-seg 0

CUDA_VISIBLE_DEVICES=1 python infer3.py \
--ckpt /data/hanchong/DiffV2IR/IR-500K/IR-500k/finetuned_checkpoints/FLIR.ckpt \
--input /data/hanchong/images-splitted-3/In-distribution/train/C14 \
--output /data/hanchong/maritime-vessel-dataset-infrared/In-distribution_IR-FLIR-2/train/C14 \
--steps=50 \
--no-seg \
--cfg-seg 0

#CUDA_VISIBLE_DEVICES=0 python infer3.py \
#--ckpt /data/hanchong/DiffV2IR/IR-500K/IR-500k/finetuned_checkpoints/FLIR.ckpt \
#--input /data/hanchong/images-splitted-3/In-distribution/train/C15 \
#--output /data/hanchong/maritime-vessel-dataset-infrared/In-distribution_IR-FLIR-2/train/C15 \
#--steps=50 \
#--no-seg \
#--cfg-seg 0

CUDA_VISIBLE_DEVICES=2 python infer3.py \
--ckpt /data/hanchong/DiffV2IR/IR-500K/IR-500k/finetuned_checkpoints/FLIR.ckpt \
--input /data/hanchong/images-splitted-3/In-distribution/train/C17 \
--output /data/hanchong/maritime-vessel-dataset-infrared/In-distribution_IR-FLIR-2/train/C17 \
--steps=50 \
--no-seg \
--cfg-seg 0

CUDA_VISIBLE_DEVICES=0 python infer3.py \
--ckpt /data/hanchong/DiffV2IR/IR-500K/IR-500k/finetuned_checkpoints/FLIR.ckpt \
--input /data/hanchong/images-splitted-3/In-distribution/train/C18 \
--output /data/hanchong/maritime-vessel-dataset-infrared/In-distribution_IR-FLIR-2/train/C18 \
--steps=50 \
--no-seg \
--cfg-seg 0

CUDA_VISIBLE_DEVICES=0 python infer3.py \
--ckpt /data/hanchong/DiffV2IR/IR-500K/IR-500k/finetuned_checkpoints/FLIR.ckpt \
--input /data/hanchong/images-splitted-3/In-distribution/train/C19 \
--output /data/hanchong/maritime-vessel-dataset-infrared/In-distribution_IR-FLIR-2/train/C19 \
--steps=50 \
--no-seg \
--cfg-seg 0

CUDA_VISIBLE_DEVICES=2 python infer3.py \
--ckpt /data/hanchong/DiffV2IR/IR-500K/IR-500k/finetuned_checkpoints/FLIR.ckpt \
--input /data/hanchong/images-splitted-3/In-distribution/train/C20 \
--output /data/hanchong/maritime-vessel-dataset-infrared/In-distribution_IR-FLIR-2/train/C20 \
--steps=50 \
--no-seg \
--cfg-seg 0