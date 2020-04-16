# python main.py \
# --algorithm mmaml \
# --model-type gatedconv \
# --condition-type affine \
# --embedding-type ConvGRU \
# --embedding-hidden-size 128 \
# --no-rnn-aggregation True \
# --fast-lr 0.05 \
# --inner-loop-grad-clip 20.0 \
# --num-updates 5 \
# --num-batches 60000 \
# --meta-batch-size 10 \
# --slow-lr 0.001 \
# --model-grad-clip 0.0 \
# --dataset multimodal_few_shot \
# --multimodal_few_shot omniglot miniimagenet cifar bird aircraft \
# --num-classes-per-batch 5 \
# --num-train-samples-per-class 5 \
# --num-val-samples-per-class 5 \
# --common-img-side-len 84 \
# --common-img-channel 3 \
# --output-folder mmaml_five_5w5s \
# --device cuda \
# --device-number 4 \
# --log-interval 50 \
# --save-interval 1000

for chk in '4000' '5000' '6000' '7000' '8000' '9000' '10000' '11000' '12000'
do
    python main.py \
    --algorithm mmaml \
    --model-type gatedconv \
    --condition-type affine \
    --embedding-type ConvGRU \
    --embedding-hidden-size 128 \
    --no-rnn-aggregation True \
    --fast-lr 0.05 \
    --inner-loop-grad-clip 20.0 \
    --num-updates 5 \
    --num-batches 50 \
    --meta-batch-size 10 \
    --slow-lr 0.001 \
    --model-grad-clip 0.0 \
    --dataset multimodal_few_shot \
    --multimodal_few_shot omniglot miniimagenet cifar bird aircraft \
    --num-classes-per-batch 5 \
    --num-train-samples-per-class 5 \
    --num-val-samples-per-class 20 \
    --common-img-side-len 84 \
    --common-img-channel 3 \
    --output-folder mmaml_five_5w5s \
    --device cuda \
    --device-number 0 \
    --log-interval 50 \
    --save-interval 1000 \
    --eval \
    --checkpoint train_dir_osc/mmaml_five_5w5s/maml_gatedconv_$chk.pt
done