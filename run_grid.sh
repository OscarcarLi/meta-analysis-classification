mkdir -p eval
python main.py \
--algorithm metaoptnet \
--model-type impregconv \
--original-conv \
--embedding-hidden-size 256 \
--no-rnn-aggregation True \
--slow-lr 0.001 \
--model-grad-clip 0. \
--dataset miniimagenet \
--num-batches-meta-train 60000 \
--num-batches-meta-val 100 \
--meta-batch-size 8 \
--num-train-samples-per-class-meta-train 5 \
--num-train-samples-per-class-meta-val 5 \
--num-train-samples-per-class-meta-test 5 \
--num-val-samples-per-class-meta-train 15 \
--num-val-samples-per-class-meta-val 15 \
--num-val-samples-per-class-meta-test 15 \
--img-side-len 84 \
--output-folder eval \
--device cuda \
--device-number 3 \
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
--optimizer adam \
--add-bias True \
--eval \
--checkpoint $1
rm -r eval



mkdir -p eval
python main.py \
--algorithm metaoptnet \
--model-type impregconv \
--original-conv \
--embedding-hidden-size 256 \
--no-rnn-aggregation True \
--slow-lr 0.001 \
--model-grad-clip 0. \
--dataset miniimagenet \
--num-batches-meta-train 60000 \
--num-batches-meta-val 100 \
--meta-batch-size 8 \
--num-train-samples-per-class-meta-train 1 \
--num-train-samples-per-class-meta-val 1 \
--num-train-samples-per-class-meta-test 1 \
--num-val-samples-per-class-meta-train 15 \
--num-val-samples-per-class-meta-val 15 \
--num-val-samples-per-class-meta-test 15 \
--img-side-len 84 \
--output-folder eval \
--device cuda \
--device-number 3 \
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
--optimizer adam \
--add-bias True \
--eval \
--checkpoint $1
rm -r eval



mkdir -p eval
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
--meta-batch-size 8 \
--num-train-samples-per-class-meta-train 5 \
--num-train-samples-per-class-meta-val 5 \
--num-train-samples-per-class-meta-test 5 \
--num-val-samples-per-class-meta-train 15 \
--num-val-samples-per-class-meta-val 15 \
--num-val-samples-per-class-meta-test 15 \
--img-side-len 84 \
--output-folder eval \
--device cuda \
--device-number 3 \
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
--optimizer adam \
--add-bias True \
--eval \
--checkpoint $1
rm -r eval



mkdir -p eval
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
--meta-batch-size 8 \
--num-train-samples-per-class-meta-train 1 \
--num-train-samples-per-class-meta-val 1 \
--num-train-samples-per-class-meta-test 1 \
--num-val-samples-per-class-meta-train 15 \
--num-val-samples-per-class-meta-val 15 \
--num-val-samples-per-class-meta-test 15 \
--img-side-len 84 \
--output-folder eval \
--device cuda \
--device-number 3 \
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
--optimizer adam \
--add-bias True \
--eval \
--checkpoint $1
rm -r eval



mkdir -p eval
python main.py \
--algorithm protonet \
--model-type impregconv \
--original-conv \
--embedding-hidden-size 256 \
--no-rnn-aggregation True \
--slow-lr 0.001 \
--model-grad-clip 0. \
--dataset miniimagenet \
--num-batches-meta-train 60000 \
--num-batches-meta-val 100 \
--meta-batch-size 8 \
--num-train-samples-per-class-meta-train 5 \
--num-train-samples-per-class-meta-val 5 \
--num-train-samples-per-class-meta-test 5 \
--num-val-samples-per-class-meta-train 15 \
--num-val-samples-per-class-meta-val 15 \
--num-val-samples-per-class-meta-test 15 \
--img-side-len 84 \
--output-folder eval \
--device cuda \
--device-number 3 \
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
--optimizer adam \
--add-bias False \
--eval \
--checkpoint $1
rm -r eval



mkdir -p eval
python main.py \
--algorithm protonet \
--model-type impregconv \
--original-conv \
--embedding-hidden-size 256 \
--no-rnn-aggregation True \
--slow-lr 0.001 \
--model-grad-clip 0. \
--dataset miniimagenet \
--num-batches-meta-train 60000 \
--num-batches-meta-val 100 \
--meta-batch-size 8 \
--num-train-samples-per-class-meta-train 1 \
--num-train-samples-per-class-meta-val 1 \
--num-train-samples-per-class-meta-test 1 \
--num-val-samples-per-class-meta-train 15 \
--num-val-samples-per-class-meta-val 15 \
--num-val-samples-per-class-meta-test 15 \
--img-side-len 84 \
--output-folder eval \
--device cuda \
--device-number 3 \
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
--optimizer adam \
--add-bias False \
--eval \
--checkpoint $1
rm -r eval