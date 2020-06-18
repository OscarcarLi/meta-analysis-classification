# bash run_grid.sh inner_solvers/minim_5w5s_SVM/maml_impregconv_20000.pt
# bash run_grid.sh inner_solvers/minim_5w1s_SVM/maml_impregconv_22000.pt
# bash run_grid.sh inner_solvers/minim_5w5s_LR/maml_impregconv_7000.pt
# bash run_grid.sh inner_solvers/minim_5w1s_LR/maml_impregconv_18000.pt
# bash run_grid.sh inner_solvers/minim_5w5s_protonet/maml_impregconv_8000.pt
# bash run_grid.sh inner_solvers/minim_5w1s_protonet/maml_impregconv_20000.pt

# export PYTHONPATH='.'
# python scripts/dump_features.py ablations/minim_5w1s_LR/maml_impregconv_24000.pt inner_solvers_features/minim_5w1s_LR_features_dict.pkl
# python scripts/dump_features.py ablations/minim_5w5s_LR/maml_impregconv_9000.pt inner_solvers_features/minim_5w5s_LR_features_dict.pkl
# python scripts/dump_features.py ablations/minim_5w15s_LR/maml_impregconv_5000.pt inner_solvers_features/minim_5w15s_LR_features_dict.pkl
# python scripts/dump_features.py ablations/minim_10w1s_LR/maml_impregconv_8000.pt inner_solvers_features/minim_10w1s_LR_features_dict.pkl

# python scripts/dump_features.py ablations/minim_5w1s_SVM/maml_impregconv_21000.pt inner_solvers_features/minim_5w1s_SVM_features_dict.pkl
# python scripts/dump_features.py ablations/minim_5w5s_SVM/maml_impregconv_20000.pt inner_solvers_features/minim_5w5s_SVM_features_dict.pkl
# python scripts/dump_features.py ablations/minim_10w1s_SVM/maml_impregconv_108000.pt inner_solvers_features/minim_10w1s_SVM_features_dict.pkl
# python scripts/dump_features.py ablations/minim_20w1s_SVM/maml_impregconv_85000.pt inner_solvers_features/minim_20w1s_SVM_features_dict.pkl


# python scripts/dump_features.py ablations/minim_5w1s_protonet/maml_impregconv_20000.pt inner_solvers_features/minim_5w1s_protonet_features_dict.pkl
# python scripts/dump_features.py ablations/minim_5w5s_protonet/maml_impregconv_8000.pt inner_solvers_features/minim_5w5s_protonet_features_dict.pkl
# python scripts/dump_features.py ablations/minim_5w15s_protonet/maml_impregconv_40000.pt inner_solvers_features/minim_5w15s_protonet_features_dict.pkl
# python scripts/dump_features.py ablations/minim_10w1s_protonet/maml_impregconv_124000.pt inner_solvers_features/minim_10w1s_protonet_features_dict.pkl
# python scripts/dump_features.py ablations/minim_10w5s_protonet/maml_impregconv_48000.pt inner_solvers_features/minim_10w5s_protonet_features_dict.pkl
# python scripts/dump_features.py ablations/minim_20w5s_protonet/maml_impregconv_15000.pt inner_solvers_features/minim_20w5s_protonet_features_dict.pkl


# LR scripts
# bash scripts_eval/eval_LR_miniimagenet.sh ablations/minim_5w1s_LR/maml_impregconv_24000.pt
# bash scripts_eval/eval_LR_miniimagenet.sh ablations/minim_5w5s_LR/maml_impregconv_9000.pt
# bash scripts_eval/eval_LR_miniimagenet.sh ablations/minim_5w15s_LR/maml_impregconv_5000.pt
# bash scripts_eval/eval_LR_miniimagenet.sh ablations/minim_10w1s_LR/maml_impregconv_8000.pt


# SVM scripts
# bash scripts_eval/eval_SVM_miniimagenet.sh ablations/minim_5w1s_SVM/maml_impregconv_21000.pt
# bash scripts_eval/eval_SVM_miniimagenet.sh ablations/minim_5w5s_SVM/maml_impregconv_20000.pt
# bash scripts_eval/eval_SVM_miniimagenet.sh ablations/minim_10w1s_SVM/maml_impregconv_108000.pt
# bash scripts_eval/eval_SVM_miniimagenet.sh ablations/minim_20w1s_SVM/maml_impregconv_85000.pt



# Protonet scripts
# bash scripts_eval/eval_protonet_miniimagenet.sh ablations/minim_5w1s_protonet/maml_impregconv_20000.pt
# bash scripts_eval/eval_protonet_miniimagenet.sh ablations/minim_5w5s_protonet/maml_impregconv_8000.pt
# bash scripts_eval/eval_protonet_miniimagenet.sh ablations/minim_5w15s_protonet/maml_impregconv_40000.pt
# bash scripts_eval/eval_protonet_miniimagenet.sh ablations/minim_10w1s_protonet/maml_impregconv_78000.pt
# bash scripts_eval/eval_protonet_miniimagenet.sh ablations/minim_10w5s_protonet/maml_impregconv_48000.pt
bash scripts_eval/eval_protonet_miniimagenet.sh ablations/minim_20w1s_protonet/maml_impregconv_8000.pt
# bash scripts_eval/eval_protonet_miniimagenet.sh ablations/minim_20w5s_protonet/maml_impregconv_15000.pt