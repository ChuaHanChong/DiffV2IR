#CUDA_VISIBLE_DEVICES=0 python infer2.py \
#--ckpt /data/hanchong/DiffV2IR/IR-500K/IR-500k/finetuned_checkpoints/FLIR.ckpt \
#--input /data/hanchong/images-splitted-3/In-distribution/test \
#--output /data/hanchong/maritime-vessel-dataset-infrared-2/In-distribution_FLIR/test \
#--steps=50 \
#--no-seg \
#--cfg-seg 0 \
#--seed 42

#CUDA_VISIBLE_DEVICES=2 python infer2.py \
#--ckpt /data/hanchong/DiffV2IR/IR-500K/IR-500k/finetuned_checkpoints/FLIR.ckpt \
#--input /data/hanchong/images-splitted-3/Out-of-distribution/test \
#--output /data/hanchong/maritime-vessel-dataset-infrared-2/Out-of-distribution_FLIR/test \
#--steps=50 \
#--no-seg \
#--cfg-seg 0 \
#--seed 42