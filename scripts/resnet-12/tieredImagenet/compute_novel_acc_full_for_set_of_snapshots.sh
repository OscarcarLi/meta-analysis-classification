#! /bin/bash

# all of the outputs (one for each snapshot) will be appended to novel_acc_variance_552.txt
# in the output_folder

n_total_classes=552

func() {
    echo "running full test_large for chkpt_0$1.pt"
    output='metal_tiered_r12_PN_n20s5q15tb1_SGD0.1Drop204050_base_correct_15:Apr:2021:01:03:17'
    # method_dataset_model_innerAlg_config_outerOpt_misc
    device='1,2'
    export PYTHONPATH='/home/oscarli/projects/meta-analysis-classification'

    python analysis/compute_novel_acc_variance.py \
    --random-seed 100 \
    --sample 0 \
    --n-chosen-classes $n_total_classes \
    --n-runs 1 \
    --preload-train False \
    --model-type resnet_12 \
    --avg-pool True \
    --projection False \
    --classifier-type no-classifier \
    --classifier-metric euclidean \
    --num-classes-train 0 \
    --algorithm-1 ProtoNet \
    --scale-factor 10. \
    --learnable-scale False \
    --dataset-path ~/projects/meta-analysis-classification/datasets/filelists/tieredImagenet-base \
    --img-side-len 84 \
    --batch-size-val 20 \
    --n-way-val 5 \
    --n-shot-val 5 \
    --n-query-val 15 \
    --n-iterations-val 500 \
    --eps 0. \
    --output-folder ${output} \
    --device-number ${device} \
    --checkpoint-1 /home/oscarli/projects/meta-analysis-classification/runs/metal_tiered_r12_PN_n20s5q15tb1_SGD0.1Drop204050_base_correct_15:Apr:2021:01:03:17/chkpt_0$1.pt
}

for snapshot in {41..60};
do func $snapshot;
done