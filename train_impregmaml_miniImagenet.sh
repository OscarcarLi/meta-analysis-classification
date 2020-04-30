python main.py \
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
--model-grad-clip 2.0 \
--dataset miniimagenet \
--num-classes-per-batch 5 \
--num-train-samples-per-class 1 \
--num-val-samples-per-class-meta-train 15 \
--num-val-samples-per-class-meta-val 15 \
--img-side-len 84 \
--output-folder impregmaml_minim_5w1s_8r_48c \
--device cuda \
--device-number 3 \
--log-interval 50 \
--save-interval 1000 \
--modulation-mat-rank 8 \
--num-channels 48 \
--val-interval 1000 \
--l2-inner-loop 0.25
# --momentum \
# --gamma-momentum 0.2 \
# --original-conv


# Nameing convention 
# dataset_type_main_model_rank_training_paradigm_embedding_model