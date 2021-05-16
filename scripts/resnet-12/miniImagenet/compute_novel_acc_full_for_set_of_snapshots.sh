#! /bin/bash

# all of the outputs (one for each snapshot) will be appended to novel_acc_variance_552.txt
# in the output_folder

n_total_classes=120
device='2,3'

output='metal_mini_r12_PN_n20s5q15tb1_SGD0.1Drop204050'

func() {
    echo "running full test_large for chkpt_0$1.pt"
    # method_dataset_model_innerAlg_config_outerOpt_misc
    export PYTHONPATH='.'

    python analysis/compute_novel_acc_variance.py \
    --sample 0 \
    --n-chosen-classes $n_total_classes \
    --n-runs 1 \
    --preload-images False \
    --algorithm-hyperparams-json-1 scripts/alg_hyperparams_jsons/mini_PN.json \
    --checkpoint-1 runs/$output/chkpt_0$1.pt \
    --dataset-path data/miniImagenet-base \
    --img-side-len 84 \
    --batch-size-val 10 \
    --n-way-val 5 \
    --n-shot-val 5 \
    --n-query-val 15 \
    --n-iterations-val 5000 \
    --output-folder ${output} \
    --device-number ${device}
}

for snapshot in {53..53};
do func $snapshot;
done