#! /bin/bash 

output='metalcvxFix0.8Var0.2_euc_MI_r12_n64_s1_q128'

nohup python main_classical.py \
--algorithm Protonet \
--model-type resnet12 \
--classifier-metric euclidean \
--projection False \
--classifier-type cvx-classifier \
--lambd 0.8 \
--img-side-len 84 \
--gamma 0. \
--lr 0.1 \
--momentum 0.9 \
--eps 0. \
--weight-decay 0.0005 \
--grad-clip 0. \
--dataset-path data/filelists/miniImagenet \
--n-epochs 200 \
--num-classes-train 64 \
--n-way-train 64 \
--fix-support \
--support-toggle 10 \
--n-shot-train 1 \
--n-query-train 0 \
--batch-size-train 128 \
--n-iterations-val 500 \
--batch-size-val 2 \
--n-way-val 5 \
--n-shot-val 1 \
--n-query-val 15 \
--output-folder ${output} \
--device cuda \
--device-number 0,1,2,3 \
--log-interval 200 \
--train-aug  > ${output}.out &
tail -f ${output}.out 

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
