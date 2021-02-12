# FixML



Training in Section 4
To train ProtoNet with Resnet-12 backbone using FixML objective on mini-imagenet, use the following scripts:
```
bash scripts/resnet-12/miniImagenet/train_5w5s_fixS_PN.sh
```
To train ProtoNet with Resnet-12 backbone using ML objective, use the following scripts:
```
bash scripts/resnet-12/miniImagenet/train_5w5s_metal_PN.sh
```

Evaluations in Section 4.2, 5.1
The evaluation script is in ```eval.py``` which would evaluate a given algorithm over a sequence of task distributions between base and novel.

Evaluations in Section 5.2, Appendix D
The variance analysis is done using python scripts in ```analysis/compute_novel_acc_variance.py``` and ```analysis/compute_base_acc_variance.py```.