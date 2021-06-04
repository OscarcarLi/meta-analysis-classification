#! /bin/bash

# all of the outputs (one for each snapshot) will be appended to novel_acc_variance_552.txt
# in the output_folder

n_total_classes=552
device='1,2'

output='metal_tiered_r12_PN_n20s5q15tb1_SGD0.1Drop204050_base_correct_15:Apr:2021:01:03:17'

func() {
    echo "running full test_large for chkpt_0$1.pt"
    # method_dataset_model_innerAlg_config_outerOpt_misc
    export PYTHONPATH='.'

    python analysis/compute_novel_acc_variance.py \
    --sample 0 \
    --n-chosen-classes $n_total_classes \
    --n-runs 1 \
    --preload-images False \
    --algorithm-hyperparams-json-1 scripts/alg_hyperparams_jsons/tiered_PN.json \
    --checkpoint-1 runs/$output/chkpt_0$1.pt
    --dataset-path data/tieredImagenet-base \
    --img-side-len 84 \
    --batch-size-val 20 \
    --n-way-val 5 \
    --n-shot-val 5 \
    --n-query-val 15 \
    --n-iterations-val 500 \
    --eps 0. \
    --output-folder ${output} \
    --device-number ${device}
}

for snapshot in {41..60};
do func $snapshot;
done