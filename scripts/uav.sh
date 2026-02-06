#!usr/bin/env bash
# 训练配置
gpu_ids=7
batch_size=32
lr=1e-3
input_img_size=256
optimizer=adamw
# linear -> poly
lr_mode=poly
num_workers=8
dataset_name=UAV
total_epoch=300
# 0.
net=lgmm
loss_fn=ce

save_path=${net}_results_UAV_epochs_${total_epoch}_lr_${lr}
CUDA_VISIBLE_DEVICES=${gpu_ids} \
python ../main.py \
    --net ${net} \
    --batch_size ${batch_size} \
    --lr ${lr} \
    --optimizer ${optimizer} \
    --lr_mode ${lr_mode} \
    --loss_fn ${loss_fn} \
    --num_workers ${num_workers} \
    --dataset_name 'UAV' \
    --total_epoch ${total_epoch} \
    --save_path ${save_path} \
    --input_img_size ${input_img_size} \
    --best_model_name 'best_model.pth'