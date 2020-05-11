python main.py \
--algorithm imp_reg_maml \
--use-label \
--model-type impregconv \
--original-conv \
--condition-type affine \
--embedding-type RegConvGRU \
--embedding-hidden-size 256 \
--no-rnn-aggregation False \
--num-batches-meta-train 60000 \
--num-batches-meta-val 100 \
--meta-batch-size 10 \
--slow-lr 0.001 \
--model-grad-clip 0. \
--dataset miniimagenet \
--num-classes-per-batch 5 \
--num-train-samples-per-class-meta-train 1 \
--num-train-samples-per-class-meta-val 1 \
--num-val-samples-per-class-meta-train 15 \
--num-val-samples-per-class-meta-val 15 \
--img-side-len 84 \
--output-folder imp_minim_5w1s_8_random_no_poison \
--device cuda \
--device-number 1 \
--log-interval 50 \
--val-interval 500 \
--save-interval 1000 \
--num-channels 64 \
--modulation-mat-rank 8 \
--l2-inner-loop 0.25 \
--modulation-mat-spec-norm 5.
# --checkpoint train_dir/imp_minim_5w1s_64_random_no_poison/maml_impregconv_5000.pt


# Nameing convention 
# dataset_type_main_model_rank_training_paradigm_embedding_model