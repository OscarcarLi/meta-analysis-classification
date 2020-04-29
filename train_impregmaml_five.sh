python main.py \
--algorithm imp_reg_maml \
--model-type impregconv \
--condition-type affine \
--embedding-type RegConvGRU \
--embedding-hidden-size 128 \
--no-rnn-aggregation True \
--fast-lr .05 \
--inner-loop-grad-clip 20.0 \
--num-updates 100 \
--num-batches-meta-train 60000 \
--num-batches-meta-val 100 \
--meta-batch-size 10 \
--slow-lr 0.001 \
--model-grad-clip 1. \
--dataset multimodal_few_shot \
--multimodal_few_shot omniglot miniimagenet cifar bird aircraft \
--num-classes-per-batch 5 \
--num-train-samples-per-class 5 \
--num-val-samples-per-class-meta-train 15 \
--num-val-samples-per-class-meta-val 15 \
--common-img-side-len 84 \
--common-img-channel 3 \
--output-folder impregmaml_five_2w5s \
--device cuda \
--device-number 1 \
--log-interval 1 \
--save-interval 1000 \
--modulation-mat-rank 32 \
--num-channels 32 \
--val-interval 1000 \
--l2-inner-loop 0.01 \
# --momentum \
# --gamma-momentum 0.7 \
# --modulation-mat-spec-norm 100. \
