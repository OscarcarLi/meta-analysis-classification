#! /bin/bash 
output='filename'
device='2,3'
randomseed=521

CUDA_VISIBLE_DEVICES="$device" nohup python main_classical.py \
--random-seed $randomseed \
--algorithm Protonet \
--model-type conv64 \
--avg-pool True \
--classifier-metric euclidean \
--projection False \
--classifier-type cvx-classifier \
--lambd 0. \
--img-side-len 84 \
--gamma 0. \
--lr 0.1 \
--momentum 0.9 \
--eps 0. \
--weight-decay 0.0005 \
--grad-clip 0. \
--dataset-path data/filelists/miniImagenet \
--n-epochs 60 \
--drop-lr-epoch 20 \
--num-classes-train 64 \
--n-way-train 64 \
--n-shot-train 5 \
--fix-support 5 \
--n-query-train 0 \
--batch-size-train 320 \
--n-iterations-val 1000 \
--n-iterations-train 1000 \
--batch-size-val 2 \
--n-way-val 5 \
--n-shot-val 5 \
--n-query-val 15 \
--output-folder ${output} \
--device cuda \
--device-number "$device" \
--log-interval 500 \
--support-aug \
--train-aug > ${output}.out &
tail -f ${output}.out

# --log-interval 200 \
# --checkpoint train_dir_2/fixS_euc_MI_r12_n64_s5_q320/classical_resnet_165.pt \
# --restart-iter 0 
# --train-aug  

# --checkpoint train_dir_2/fixS_MI_r12_n64_s3_q320_cos/classical_resnet_080.pt \
# --load-optimizer \
# --restart-iter 80 \

# --checkpoint train_dir/metal_FC100_r12_n60_s5_q128_cvx0.8/classical_resnet_055.pt \
# --load-optimizer
# --lowdim 512 \


# python main_classical.py \
# --algorithm ProtonetCosine \
# --model-type resnet \
# --classifier-type distance-classifier \
# --img-side-len 224 \
# --gamma 0. \
# --lr 0.001 \
# --weight-decay 0.0005 \
# --grad-clip 0. \
# --dataset-path data/filelists/CUB \
# --n-epochs 400 \
# --num-classes-train 100 \
# --batch-size-train 64 \
# --n-iterations-val 50 \
# --batch-size-val 1 \
# --n-way-val 5 \
# --n-shot-val 5 \
# --n-query-val 15 \
# --output-folder classical_CUB_dc_bn \
# --device cuda \
# --device-number 0,1,2,3 \
# --log-interval 30 \
# --train-aug
