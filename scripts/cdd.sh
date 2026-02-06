#!usr/bin/env bash
# 训练配置
gpu_ids=4
batch_size=16
lr=1e-3
optimizer=adamw
# linear -> poly
lr_mode=poly
num_workers=8
dataset_name=CDD
total_epoch=300
# 0.
net=lgmm
#1.
#net=baseline
#2.
#net=baseline+dlgpe
#3.
#net=baseline+pmm
#4.
#net=lgmm_with_resnet18
#5.
#net=lgmm_with_elgca
loss_fn=ce

save_path=${net}_results_${dataset_name}_epochs_${total_epoch}_lr_${lr}
CUDA_VISIBLE_DEVICES=${gpu_ids} \
python ../main.py \
    --net ${net} \
    --batch_size ${batch_size} \
    --lr ${lr} \
    --optimizer ${optimizer} \
    --lr_mode ${lr_mode} \
    --loss_fn ${loss_fn} \
    --num_workers ${num_workers} \
    --dataset_name ${dataset_name} \
    --total_epoch ${total_epoch} \
    --save_path ${save_path}