python main.py \
--algorithm protosvm \
--model-type impregconv \
--original-conv \
--slow-lr 0.1 \
--model-grad-clip 0. \
--dataset miniimagenet \
--num-batches-meta-train 60000 \
--num-batches-meta-val 100 \
--meta-batch-size 8 \
--num-classes-per-batch-meta-train 5 \
--num-classes-per-batch-meta-val 5 \
--num-classes-per-batch-meta-test 5 \
--num-train-samples-per-class-meta-train 5 \
--num-train-samples-per-class-meta-val 5 \
--num-train-samples-per-class-meta-test 5 \
--num-val-samples-per-class-meta-train 15 \
--num-val-samples-per-class-meta-val 15 \
--num-val-samples-per-class-meta-test 15 \
--img-side-len 84 \
--output-folder minim_5w5s_protosvm \
--device cuda \
--device-number 2 \
--log-interval 50 \
--save-interval 1000 \
--val-interval 1000 \
--num-channels 64 \
--verbose True \
--retain-activation True \
--use-group-norm True \
--optimizer sgd \
--add-bias False

# --checkpoint train_dir/impregmaml_minim_5w5s_metaoptnet_fixed/maml_impregconv_20000.pt


# --momentum \
# --gamma-momentum 0.2 \
# --modulation-mat-rank 8 \


# Nameing convention 
# dataset_type_main_model_rank_training_paradigm_embedding_model   