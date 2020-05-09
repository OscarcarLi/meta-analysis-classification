python main.py \
--use-tboard \
--algorithm imp_reg_maml \
--model-type impregconv \
--condition-type affine \
--embedding-type RegConvGRU \
--embedding-hidden-size 256 \
--no-rnn-aggregation True \
--fast-lr .03 \
--inner-loop-grad-clip 0. \
--num-updates 5 \
--num-batches-meta-train 60000 \
--num-batches-meta-val 100 \
--meta-batch-size 10 \
--slow-lr 0.001 \
--slow-lr-embedding-model 0.001 \
--model-grad-clip 5. \
--dataset miniimagenet \
--num-classes-per-batch-meta-train 5 \
--num-classes-per-batch-meta-val 5 \
--num-train-samples-per-class 1 \
--num-val-samples-per-class-meta-train 15 \
--num-val-samples-per-class-meta-val 15 \
--img-side-len 84 \
--output-folder impregmaml_minim_5w1s_32 \
--device cuda \
--device-number 3 \
--log-interval 50 \
--save-interval 500 \
--modulation-mat-rank 32 \
--num-channels 32 \
--num-channels-embedding-model 32 \
--val-interval 500 \
--l2-inner-loop 0.25 \
--modulation-mat-spec-norm 1. \
--conv-embedding-avgpool-after-conv False \
--original-conv \
--use-label \
--verbose False \
# --checkpoint train_dir/impregmaml_minim_5w1s_3/maml_impregconv_2000.pt
# --eval

# --randomize-modulation-mat \
# --normalize-norm 0. \
# --eval \
# --checkpoint train_dir/impregmaml_minim_5w1s_normalize_2/maml_impregconv_19500.pt

# --randomize-modulation-mat \
# --randomize-modulation-mat \
# --eye-modulation-mat \
# --tie-conv-embedding-model \
# --checkpoint train_dir/impregmaml_minim_5w1s_103/maml_impregconv_13000.pt
# --momentum \
# --gamma-momentum 0.2 \

# lower lambda means higher ub for hessian inv (1/x)
# lower spec norm means higher lb for hessian inv (1/x^2)
# lambda -> (sn^2 + lambda)    [hessian]
# 1./(sn^2 + lambda) -> 1./lambda     [hessian inv]

# Naming convention 
# dataset_type_main_model_rank_training_paradigm_embedding_model