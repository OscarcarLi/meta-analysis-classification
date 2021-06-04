# Two Sides of Meta-Learning Evaluation: In vs. Out of Distribution

## Modified FSL Datasets Creation

Download the following (modified) OOD FSL datasets from Anonymous Google Drive (using gdown) into the "data" folder

```
gdown --id 1wUvTVEhZorao9DHgCQZySejHGhI9JdZw -O tieredImagenet-base.tar
gdown --id 12tRzxlnWSMd3j-9D3w35VjXYfmVYenmR -O cifar-fs-base.tar
gdown --id 11pRhOK9HFZFjbdqnJYNzfRM4-FSLYS1g -O FC100-base.tar
gdown --id 1oAvvVIemrTrljJ6Qd5LFGGe0MXRLdtFU -O miniImagenet-base.tar
```

Once the FSL datasets are downloaded and uncompressed into 4 folders (one for each dataset), process each file according to the readme in each folder to fully process the datasets as mentioned in Section 4 of the main paper. The combined set of instructions for this part can also be found in dataset_setup.md file.


## Zappos-ID/OOD and FEMNIST dataset creation

### Zappos-ID/OOD

1. Download the UT Zappos-50k corpus from http://vision.cs.utexas.edu/projects/finegrained/utzap50k/. Specifically download the zip file into ut-zap50k-images-square.zip and the metadata into ut-zap50k-data.zip.
2. Process the dataset using the notebook fed_data/zappos/zap50k-json-creation-any-negative-id-ood.ipynb. Within the notebook set the ZAPPOS_ROOT to the appropriate directory where contents from UT Zappos-50k were downloaded and unzipped. 
3. Create train/val/test jsons using the notebook fed_data/zappos/create_json-id-ood.ipynb

### FEMNIST

1. Download the FEMNIST raw_data to the folder fed_data/femnist/data, using the following command.
```
gdown --id 1VSpjUTy9XNC2lRkxWO3Lc6qtDMvzhph_ -O femnist.tar
``` 

2. Run the notebook in fed_data/femnist/create_json_drop_classes_fixedsq_split.ipynb after setting the appropriate FEMNIST_ROOT with the path to the downloaded raw_data. 

## Training

### Training on modified FSL datasets

To train PN on miniImagenet-mod run:
```
bash scripts/resnet-12/miniImagenet/train_5w5s_metal_PN.sh
```
To train PN on miniImagenet-mod with fixml sampling run:
```
bash scripts/resnet-12/miniImagenet/train_5w5s_fixS_PN.sh
```

Scripts to train PN, Ridge, SVM and FO-MAML can be found in scripts/resnet-12/miniImagenet for mini-M, scripts/resnet-12/tieredImagenet for tiered-M, scripts/resnet-12/FC100-base for FC-M and scripts/resnet-12/cifar-fs-base for cifar-M. 


### Training on Zappos-ID

Scripts to train on Zappos-ID can be found in scripts/resnet-12/zappos. For OOD evaluation seperate scripts are present in the same folder.


### Training on FEMNIST

Scripts to train and evaluate ID performance on FEMNIST can be found in scripts/conv-64/femnist. 


## Compute Kendall Rank Coefficient
1. Download the validation and test csvs for a given run using the tensorboard
2. Run the file in analysis/compute_kendall_rank_coefficient.py

## Compute Conclusion Flips and Exaggerations

1. Have PN and Ridge trained model trajectories on mini-M.
2. Run scripts/resnet-12/miniImagenet/compute_novel_acc_full_for_set_of_snapshots.sh for each of the trajectories. The result (performance on large underlying set) will be logged in the respective trajectory's runs folder novel_acc_variance_*.txt.
3. Identify a pair of algorithm snapshot (one from each trajectory) whose performance difference on the underlying set >= 0.5%.
4. Run scripts/resnet-12/miniImagenet/compute_novel_acc_variance_PN_Ridge.sh to check for conclusion flip. Can check when difference number of subset classes are chosen, e.g. 20, 40, 80, 160, etc.


