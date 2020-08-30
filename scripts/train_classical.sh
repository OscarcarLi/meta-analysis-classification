python main_classical.py \
--algorithm Protonet \
--model-type resnet12 \
--classifier-type avg-classifier \
--lambd 0.5 \
--img-side-len 84 \
--gamma 0. \
--lr 0.1 \
--momentum 0.9 \
--eps 0. \
--weight-decay 0.0005 \
--grad-clip 0. \
--dataset-path data/filelists/CUB \
--n-epochs 60 \
--num-classes-train 100 \
--n-way-train 100 \
--n-shot-train 3 \
--n-query-train 3 \
--batch-size-train 128 \
--n-iterations-val 400 \
--batch-size-val 1 \
--n-way-val 5 \
--n-shot-val 5 \
--n-query-val 15 \
--output-folder metal_CUB_r12_n100_s3_q3_euc_cvxfixSnofixS0.5 \
--device cuda \
--device-number 0,1,2,3 \
--log-interval 250 \
--train-aug \
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
