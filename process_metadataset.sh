#! /bin/bash

# omniglot
python process_metadataset.py --base-path data/meta_dataset/mds_tfrecords/ --dataset omniglot --split base
python process_metadataset.py --base-path data/meta_dataset/mds_tfrecords/ --dataset omniglot --split val
python process_metadataset.py --base-path data/meta_dataset/mds_tfrecords/ --dataset omniglot --split novel

# aircraft
python process_metadataset.py --base-path data/meta_dataset/mds_tfrecords/ --dataset aircraft --split base
python process_metadataset.py --base-path data/meta_dataset/mds_tfrecords/ --dataset aircraft --split val
python process_metadataset.py --base-path data/meta_dataset/mds_tfrecords/ --dataset aircraft --split novel

# cu_birds
python process_metadataset.py --base-path data/meta_dataset/mds_tfrecords/ --dataset cu_birds --split base
python process_metadataset.py --base-path data/meta_dataset/mds_tfrecords/ --dataset cu_birds --split val
python process_metadataset.py --base-path data/meta_dataset/mds_tfrecords/ --dataset cu_birds --split novel

# dtd
python process_metadataset.py --base-path data/meta_dataset/mds_tfrecords/ --dataset dtd --split base
python process_metadataset.py --base-path data/meta_dataset/mds_tfrecords/ --dataset dtd --split val
python process_metadataset.py --base-path data/meta_dataset/mds_tfrecords/ --dataset dtd --split novel

# fungi
python process_metadataset.py --base-path data/meta_dataset/mds_tfrecords/ --dataset fungi --split base
python process_metadataset.py --base-path data/meta_dataset/mds_tfrecords/ --dataset fungi --split val
python process_metadataset.py --base-path data/meta_dataset/mds_tfrecords/ --dataset fungi --split novel

# traffic_sign
python process_metadataset.py --base-path data/meta_dataset/mds_tfrecords/ --dataset traffic_sign --split base
python process_metadataset.py --base-path data/meta_dataset/mds_tfrecords/ --dataset traffic_sign --split val
python process_metadataset.py --base-path data/meta_dataset/mds_tfrecords/ --dataset traffic_sign --split novel


# vgg_flower
python process_metadataset.py --base-path data/meta_dataset/mds_tfrecords/ --dataset vgg_flower --split base
python process_metadataset.py --base-path data/meta_dataset/mds_tfrecords/ --dataset vgg_flower --split val
python process_metadataset.py --base-path data/meta_dataset/mds_tfrecords/ --dataset vgg_flower --split novel


# ilsvrc_2012
python process_metadataset.py --base-path data/meta_dataset/mds_tfrecords/ --dataset ilsvrc_2012 --split base
python process_metadataset.py --base-path data/meta_dataset/mds_tfrecords/ --dataset ilsvrc_2012 --split val
python process_metadataset.py --base-path data/meta_dataset/mds_tfrecords/ --dataset ilsvrc_2012 --split novel

# mscoco
python process_metadataset.py --base-path data/meta_dataset/mds_tfrecords/ --dataset mscoco --split base
python process_metadataset.py --base-path data/meta_dataset/mds_tfrecords/ --dataset mscoco --split val
python process_metadataset.py --base-path data/meta_dataset/mds_tfrecords/ --dataset mscoco --split novel


# quickdraw
python process_metadataset.py --base-path data/meta_dataset/mds_tfrecords/ --dataset quickdraw --split base
python process_metadataset.py --base-path data/meta_dataset/mds_tfrecords/ --dataset quickdraw --split val
python process_metadataset.py --base-path data/meta_dataset/mds_tfrecords/ --dataset quickdraw --split novel
