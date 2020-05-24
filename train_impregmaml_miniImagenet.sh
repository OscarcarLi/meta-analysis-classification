python main.py \
--algorithm imp_reg_maml \
--model-type impregconv \
--original-conv \
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
--output-folder imp_minim_5w1s_random_bootstrap_l1.0_r5_n32 \
--device cuda \
--device-number 1 \
--log-interval 50 \
--val-interval 500 \
--save-interval 100 \
--num-channels 64 \
--modulation-mat-rank 5 \
--l2-inner-loop 1. \
--n-projections 32 \
--modulation-mat-spec-norm 0.
# --checkpoint train_dir/imp_minim_5w5s_bootstrap_l10.0/maml_impregconv_14000.pt \
# --eval
# --inverse-hessian True
# --eval


# --embedding-type RegConvGRU \
# --conv-embedding-avgpool-after-conv False \
# --embedding-hidden-size 256 \
# --no-rnn-aggregation True \


# Nameing convention 
# dataset_type_main_model_rank_training_paradigm_embedding_model