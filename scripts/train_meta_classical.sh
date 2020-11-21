#! /bin/bash
output='fixS_MI_conv48_n5_s5_q15_qp15_bs2_sgdMAML_alpha0.01_metaLR0.001_rs10725'
device='2'

CUDA_VISIBLE_DEVICES="$device" nohup python main_meta_classical.py \
--random-seed 10725 \
--algorithm InitBasedAlgorithm \
--init-meta-algorithm MAML \
--inner-update-method sgd \
--num-updates-inner-train 5 \
--num-updates-inner-val 5 \
--alpha 0.01 \
--model-type conv48 \
--classifier-type linear \
--avg-pool "False" \
--classifier-metric euclidean \
--projection "False" \
--img-side-len 84 \
--lr 0.001 \
--eps 0. \
--weight-decay 0.0 \
--grad-clip 10. \
--dataset-path data/filelists/miniImagenet \
--n-epochs 100 \
--n-iters-per-epoch 500 \
--drop-lr-epoch 101 \
--num-classes-train 5 \
--batch-size-train 2 \
--n-way-train 5 \
--n-shot-train 5 \
--fix-support 5 \
--n-query-train 15 \
--n-query-pool 15 \
--batch-size-val 1 \
--n-way-val 5 \
--n-shot-val 5 \
--n-query-val 15 \
--n-iterations-val 100 \
--output-folder ${output} \
--device cuda \
--device-number "$device" \
--log-interval 100 > ${output}.out &
tail -f ${output}.out

# --checkpoint train_dir_2/metal_conv64_n5_s5_q15_qp15_bs4_sgdFOMAML_nodrop_rs10725_nosupportaug_noqueryaug/classical_resnet_020.pt \
# --checkpoint train_dir_2/metal_conv64_n5_s5_q15_qp15_bs4_sgdReptile_nodrop_rs10725_nosupportaug_noqueryaug/classical_resnet_001.pt \
# --checkpoint train_dir_2/fixS5_conv64_r12_n20_s5_q15_qp15_bs1_euc_drop20_rs10725/classical_resnet_060.pt \
# --checkpoint train_dir_2/metal_MI_r12_n64_s5_q5_qp5_bs1_euc_metaoptdataaug_drop20/classical_resnet_024.pt \
# --checkpoint train_dir_2/fixS1_cifar_r12_n5_s1_q15_qp50_bs8_euc_metaoptdataaug_sansgap_supportaug_drop20_run1/classical_resnet_020.pt \
# --restart-iter 0 \
# --checkpoint train_dir_2/fixS_cifar_r12_n64_s5_q320_euc/classical_resnet_118.pt \
# --restart-iter 0 \
# --checkpoint train_dir_2/fixS5_cifar_r12_n64_s5_q15_euc/classical_resnet_009.pt \
# --restart-iter 0 \
# --train-aug > ${output}.out &
# --fix-support 10 \
# --checkpoint train_dir_2/fixS_euc_MI_r12_n64_s5_q320/classical_resnet_150.pt \
# --restart-iter 0
# train-aug
# --checkpoint train_dir/classical_miniimagenet_avg_classifier_5/classical_resnet_163.pt
# --restart-iter 150
