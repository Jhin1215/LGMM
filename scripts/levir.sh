#!usr/bin/env bash
# 训练配置
gpu_ids=7
#batch_size=1
batch_size=32
lr=1e-3
optimizer=adamw
lr_mode=poly
num_workers=8
dataset_name=LEVIR-CD
total_epoch=300
loss_fn=ce
input_img_size=256
#0.
net=lgmm
# ban encoder
#net=ablation_enc
#1.
#net=baseline
#2.
#net=baseline+dlgpe
#3.
#net=baseline+pmm
#4.
#net=lgmm_with_resnet18
#5.
#net=lgmm_with_elgc_decoder
#6.
#net=lgmm_without_multi_conv
#7.
#net=lgmm_without_w_attn
#8.
#net=lgmm_without_sw_attn
# 9.
#net=lgmm_only_multi_conv
# 10.
#net=lgmm_only_w_attn
# 11.
#net=lgmm_only_sw_attn
#12.
#net=lgmm_with_ch_split
#13.
#net=lgmm_multi_conv_n1
#13.
#net=lgmm_multi_conv_n2
#14.
#net=lgmm_multi_conv_n3
#15.
#net=lgmm_attn_v1
#16.
#net=lgmm_attn_v2
#17.
#net=lgmm_attn_v3
#18.
#net=lgmm_attn_v4

#18.
#net=lgmm_without_dynamic_residual
#19.
#net=lgmm_without_mask_init
#20.
#net=lgmm_pmm_n1
#21.
#net=lgmm_pmm_n2
#22.
#net=lgmm_pmm_n3
#23.
#net=lgmm_pmm_rot0
#24.
#net=lgmm_pmm_rot90
#25.
#net=lgmm_pmm_rot270
#26.
#net=lgmm_pmm_2_dirs
#net=lgmm_8_dirs
#net=lgmm_without_pyramid

save_path=${net}_results_LEVIR-CD_epochs_${total_epoch}_lr_${lr}
CUDA_VISIBLE_DEVICES=${gpu_ids} \
python ../main.py \
    --net ${net} \
    --batch_size ${batch_size} \
    --lr ${lr} \
    --optimizer ${optimizer} \
    --lr_mode ${lr_mode} \
    --loss_fn ${loss_fn} \
    --num_workers ${num_workers} \
    --dataset_name 'LEVIR' \
    --total_epoch ${total_epoch} \
    --save_path ${save_path} \
    --input_img_size ${input_img_size} \
    --best_model_name 'best_model.pth'