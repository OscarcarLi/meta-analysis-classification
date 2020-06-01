python main.py \
--algorithm imp_reg_maml \
--model-type impregconv \
--original-conv \
--embedding-hidden-size 256 \
--no-rnn-aggregation True \
--slow-lr 0.001 \
--model-grad-clip 0. \
--dataset miniimagenet \
--num-batches-meta-train 60000 \
--num-batches-meta-val 100 \
--meta-batch-size 10 \
--num-train-samples-per-class-meta-train 1 \
--num-val-samples-per-class-meta-train 15 \
--num-train-samples-per-class-meta-val 1 \
--num-val-samples-per-class-meta-val 15 \
--img-side-len 84 \
--output-folder impregmaml_minim_5w1s_r2d2_fixed \
--device cuda \
--device-number 1 \
--log-interval 50 \
--save-interval 1000 \
--val-interval 1000 \
--num-channels 64 \
--original-conv \
--l2-inner-loop 10.0 \
--hessian-inverse True \
--no-modulation True \
--verbose True \
--retain-activation True \
--use-group-norm True \
--optimizer adam


# --momentum \
# --gamma-momentum 0.2 \
# --modulation-mat-rank 8 \


# Nameing convention 
# dataset_type_main_model_rank_training_paradigm_embedding_model