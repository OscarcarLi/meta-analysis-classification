python main.py \
--algorithm imp_reg_maml \
--model-type impregconv \
--original-conv \
--embedding-type RegConvGRU \
--conv-embedding-avgpool-after-conv False \
--use-label \
--embedding-hidden-size 128 \
--no-rnn-aggregation True \
--num-batches-meta-train 60000 \
--num-batches-meta-val 100 \
--meta-batch-size 10 \
--slow-lr 0.0001 \
--model-grad-clip 0. \
--dataset miniimagenet \
--num-classes-per-batch 5 \
--num-train-samples-per-class-meta-train 1 \
--num-train-samples-per-class-meta-val 1 \
--num-val-samples-per-class-meta-train 15 \
--num-val-samples-per-class-meta-val 15 \
--img-side-len 84 \
--output-folder imp_minim_5w1s_d71 \
--device cuda \
--device-number 3 \
--log-interval 50 \
--val-interval 500 \
--save-interval 1000 \
--num-channels 64 \
--modulation-mat-rank 64 \
--l2-inner-loop .1 \
--modulation-mat-spec-norm 0.
# --checkpoint train_dir/imp_minim_5w1s_d70/maml_impregconv_2000.pt \
# --eval
# --inverse-hessian True
# --eval

# Nameing convention 
# dataset_type_main_model_rank_training_paradigm_embedding_model