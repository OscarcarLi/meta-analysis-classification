python main.py \
--algorithm reg_maml \
--model-type regconv \
--condition-type affine \
--embedding-type RegConvGRU \
--embedding-hidden-size 128 \
--no-rnn-aggregation False \
--fast-lr .5 \
--inner-loop-grad-clip 20.0 \
--num-updates 5 \
--num-batches-meta-train 60000 \
--num-batches-meta-val 100 \
--meta-batch-size 10 \
--slow-lr 0.001 \
--model-grad-clip 5.0 \
--dataset miniimagenet \
--num-classes-per-batch 5 \
--num-train-samples-per-class 1 \
--num-val-samples-per-class-meta-train 15 \
--num-val-samples-per-class-meta-val 15 \
--img-side-len 84 \
--output-folder regmaml_minim_5w1s_32chorig_128r_momentum_rnn \
--device cuda \
--device-number 2 \
--log-interval 50 \
--save-interval 1000 \
--modulation-mat-rank 128 \
--num-channels 32 \
--val-interval 1000 \
--momentum \
--gamma-momentum 0.2 \
--original-conv


# Nameing convention 
# dataset_type_main_model_rank_training_paradigm_embedding_model