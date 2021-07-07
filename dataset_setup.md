How to setup the datasets
- tieredImagenet-base
    - cd data/
    - download tieredImagenet-base.tar into the folder data/
    gdown --id 1wUvTVEhZorao9DHgCQZySejHGhI9JdZw -O tieredImagenet-base.tar
    - tar xf tieredImagenet-base.tar
    - this will extract out a tieredImagenet-base folder
    - cd tieredImagenet-base
    - python3 make_json_base_val_novel_base_test_novel_large.py

- miniImagenet-base
    - cd data/
    - download miniImagenet-base.tar into the folder data/
        gdown --id 1oAvvVIemrTrljJ6Qd5LFGGe0MXRLdtFU -O miniImagenet-base.tar
    - tar xf miniImagenet-base.tar
    - cd miniImagenet-base
    - python3 make_json_base_val_novel_base_test_novel_large.py

- FC100-base
    - cd data/
    - download FC100-base.tar into the folder data/
        gdown --id 11pRhOK9HFZFjbdqnJYNzfRM4-FSLYS1g -O FC100-base.tar
    - tar xf FC100-base.tar
    - cd FC100-base
    - python3 process.py
    - python3 make_json_base_val_novel_base_test.py

- cifar-fs-base
    - cd data
    - download cifar-fs-base.tar into the folder data/
        gdown --id 12tRzxlnWSMd3j-9D3w35VjXYfmVYenmR -O cifar-fs-base.tar
    - tar xf cifar-fs-base.tar
    - cd cifar-fs-base
    - bash make_cifar-fs-base.sh



- meta-dataset

Main Setup

```bash
git clone https://github.com/google-research/meta-dataset.git
cd meta-dataset 
export ROOT=$(pwd)
export PYTHONPATH='.'
mkdir -p data
mkdir -p data/meta_dataset
mkdir -p data/meta_dataset/mds_root
mkdir -p data/meta_dataset/mds_splits
mkdir -p data/meta_dataset/mds_tfrecords
export DATASRC=$(pwd)/data/meta_dataset/mds_root
export SPLITS=$(pwd)/data/meta_dataset/mds_splits
export RECORDS=$(pwd)/data/meta_dataset/mds_tfrecords
```

 - 1. Omniglot

```bash

mkdir $DATASRC/omniglot
cd $DATASRC/omniglot
wget https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip
wget https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip
unzip images_background.zip
unzip images_evaluation.zip
cd $ROOT

python -m meta_dataset.dataset_conversion.convert_datasets_to_records \
  --dataset=omniglot \
  --omniglot_data_root=$DATASRC/omniglot \
  --splits_root=$SPLITS \
  --records_root=$RECORDS
```

- 2.  Aircraft

```bash
cd $DATASRC
wget http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz
tar -xvzf fgvc-aircraft-2013b.tar.gz
rm fgvc-aircraft-2013b.tar.gz
cd $ROOT

python -m meta_dataset.dataset_conversion.convert_datasets_to_records \
  --dataset=aircraft \
  --aircraft_data_root=$DATASRC/fgvc-aircraft-2013b \
  --splits_root=$SPLITS \
  --records_root=$RECORDS
```

- 3. CUB

```bash
cd $DATASRC
gdown --id 1hbzc_P1FuxMkcabkgn9ZKinBwW683j45
tar -xvzf CUB_200_2011.tgz
rm CUB_200_2011.tgz
cd $ROOT

python -m meta_dataset.dataset_conversion.convert_datasets_to_records \
  --dataset=cu_birds \
  --cu_birds_data_root=$DATASRC/CUB_200_2011 \
  --splits_root=$SPLITS \
  --records_root=$RECORDS
```

- 4. DTD

```bash
cd $DATASRC
wget https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz
tar -xvzf dtd-r1.0.1.tar.gz
rm dtd-r1.0.1.tar.gz
cd $ROOT

python -m meta_dataset.dataset_conversion.convert_datasets_to_records \
  --dataset=dtd \
  --dtd_data_root=$DATASRC/dtd \
  --splits_root=$SPLITS \
  --records_root=$RECORDS
```

- 5. fungi

```bash
mkdir $DATASRC/fungi
cd $DATASRC/fungi
wget https://labs.gbif.org/fgvcx/2018/fungi_train_val.tgz
wget https://labs.gbif.org/fgvcx/2018/train_val_annotations.tgz
tar -xvzf fungi_train_val.tgz
tar -xvzf https://labs.gbif.org/fgvcx/2018/train_val_annotations.tgz
cd $ROOT

python -m meta_dataset.dataset_conversion.convert_datasets_to_records \
  --dataset=fungi \
  --fungi_data_root=$DATASRC/fungi \
  --splits_root=$SPLITS \
  --records_root=$RECORDS
```

- 6. vgg_flower

```bash
mkdir $DATASRC/vgg_flower
cd $DATASRC/vgg_flower
wget http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz
wget http://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat
tar -xvzf 102flowers.tgz
cd $ROOT

python -m meta_dataset.dataset_conversion.convert_datasets_to_records \
  --dataset=vgg_flower \
  --vgg_flower_data_root=$DATASRC/vgg_flower \
  --splits_root=$SPLITS \
  --records_root=$RECORDS
```

- 7. traffic_sign

```bash
cd $DATASRC
wget https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip
unzip GTSRB_Final_Training_Images.zip
rm GTSRB_Final_Training_Images.zip
cd $ROOT

python -m meta_dataset.dataset_conversion.convert_datasets_to_records \
  --dataset=traffic_sign \
  --traffic_sign_data_root=$DATASRC/GTSRB \
  --splits_root=$SPLITS \
  --records_root=$RECORDS
```

- 8. mscoco

```bash
mkdir $DATASRC/mscoco
cd $DATASRC/mscoco/ mkdir train2017
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip train2017.zip
unzip annotations_trainval2017.zip
mv annotations/* ./
rm -r annotations
cd $ROOT

python -m meta_dataset.dataset_conversion.convert_datasets_to_records \
  --dataset=mscoco \
  --mscoco_data_root=$DATASRC/mscoco \
  --splits_root=$SPLITS \
  --records_root=$RECORDS
```

- 9. ilsvrc_2012

Download ilsvrc2012_img_train.tar, from the ILSVRC2012 website

Extract it into ILSVRC2012_img_train/ at $DATASRC, which should contain 1000 files, named n????????.tar

Extract each of `ILSVRC2012_img_train/n????????.tar` in its own directory (expected time: ~30 minutes), for instance:

```bash
cd ILSVRC2012_img_train
for FILE in n*.tar;
do
  mkdir ${FILE/.tar/};
  cd ${FILE/.tar/};
  tar xvf ../$FILE;
  cd ..;
done
wget http://www.image-net.org/data/wordnet.is_a.txt
wget http://www.image-net.org/data/words.txt
cd $ROOT

python -m meta_dataset.dataset_conversion.convert_datasets_to_records \
  --dataset=ilsvrc_2012 \
  --ilsvrc_2012_data_root=$DATASRC/ILSVRC2012_img_train \
  --splits_root=$SPLITS \
  --records_root=$RECORDSx
```

- 10. QuickDraw

```bash
mkdir $DATASRC/quickdraw
gsutil -m cp gs://quickdraw_dataset/full/numpy_bitmap/*.npy $DATASRC/quickdraw
cd $ROOT

python -m meta_dataset.dataset_conversion.convert_datasets_to_records \
  --dataset=quickdraw \
  --quickdraw_data_root=$DATASRC/quickdraw \
  --splits_root=$SPLITS \
  --records_root=$RECORDS
```