import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import argparse
import gin
import os
import json
import PIL
import numpy as np
import tqdm
import h5py
import matplotlib.pyplot as plt
from collections import Counter
import functools
from meta_dataset.data import reader
from meta_dataset.data import config
from meta_dataset.data import dataset_spec as dataset_spec_lib
from meta_dataset.data import learning_spec
from meta_dataset.data import pipeline
import glob
from meta_dataset.data.pipeline import process_batch
GIN_FILE_PATH = 'meta_dataset/learn/gin/setups/data_config.gin'
gin.parse_config_file(GIN_FILE_PATH)


LEARNING_SPEC_MAP = {
    'base':learning_spec.Split.TRAIN, 
    'val':learning_spec.Split.VALID, 
    'novel':learning_spec.Split.TEST
}


def store_single_hdf5(imagepath, image):
    """ Stores a single image to an HDF5 file.
        Parameters:
        ---------------
        imagepath   path to image
        image       image array, (32, 32, 3) to be stored
    """
    # Create a new HDF5 file
    file = h5py.File(imagepath, "a")

    # Create a dataset in the file (uint8)
    file.create_dataset(
        "image", np.shape(image), h5py.h5t.STD_U8BE, data=image
    )
    
    file.close()



def get_dataset(dataset_path, split, batch_size, image_size):

    # loads dataset specifications like classes in each split
    dataset_spec = dataset_spec_lib.load_dataset_spec(dataset_path)
    batch_reader = reader.BatchReader(
        dataset_spec=dataset_spec, 
        split=LEARNING_SPEC_MAP[split], 
        shuffle_buffer_size=0,
        read_buffer_size_bytes=1000, 
        num_prefetch=0)

    # fetch a list of datasets, one for each class
    class_datasets = batch_reader.construct_class_datasets(repeat=False, shuffle=False)
    num_classes = len(class_datasets)
    
    # zero index the class set
    start_ind = batch_reader.class_set[0]
    class_set = [
        batch_reader.class_set[ds_id] - start_ind for ds_id in range(num_classes)
    ]
    if list(class_set) != list(range(num_classes)):
        raise NotImplementedError('Batch training currently assumes the class '
                            'set is contiguous and starts at 0.')
  
    # create tf dataset that samples from each class dataset
    dataset = tf.data.experimental.sample_from_datasets(
        class_datasets)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(1)

    # convert example strings to actual tensors
    map_fn = functools.partial(process_batch, image_size=image_size)
    dataset = dataset.map(map_fn)

    # total classes and images
    classes = dataset_spec.get_classes(LEARNING_SPEC_MAP[split])
    total_images = sum([
        v for k, v in dataset_spec.to_dict()['images_per_class'].items() if k in classes])
    print(f"Total classes in split: {len(classes)}, Total images in split: ", total_images)

    return dataset, total_images


def process_dataset(dataset, images_path, total_images, batch_size):
    
    
    # final json dataset that is returned
    json_dataset = {
        'label_names': [],
        'image_names': [],
        'image_labels': []
    }

    # classes in split, total images in split, image counter
    # if images_path already has images, then we need to increment from last image
    existing_images = sorted(glob.glob(f'{images_path}/img_*.h5'))
    if not existing_images:
        counter = 0
    else:
        last_count = int(os.path.basename(
            existing_images[-1]).split('_')[1].split('.')[0])
        counter = last_count
    print("Starting counter from", counter+1)

    # iterate over the dataset and dump the images in the image directory
    for im_batch, label_batch  in tqdm.tqdm(dataset, total=int(total_images/batch_size)):
        # convert to numpy
        im_batch = im_batch.numpy()
        label_batch = label_batch.numpy()
        
        # convert images from [-1,1] -> [0,1] -> [0, 255] uint8
        im_batch = ((im_batch / 2 + 0.5) * 255).astype(np.uint8)
        
        # loop over images in batch
        for im, label in zip(im_batch, label_batch):
            im_path = os.path.join(images_path, "img_{:07d}".format(counter + 1) + ".h5") # image path
            json_dataset['image_names'].append(im_path)
            json_dataset['image_labels'].append(int(label))
            store_single_hdf5(im_path, im)
            counter += 1 # increment counter
    
    # label names are just unique list of image labels
    json_dataset['label_names'] = list(set(json_dataset['image_labels']))
    print("Created json with ...")
    print(Counter(json_dataset['image_labels']))

    return json_dataset



if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Process metadatasets to create json.')
    
    parser.add_argument('--base-path', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--split', type=str, required=True)

    parser.add_argument('--image-size', type=int, default=84)
    parser.add_argument('--batch-size', type=int, default=100)
    
         
    args = parser.parse_args()

    # dataset path
    dataset_path = os.path.join(args.base_path, args.dataset)
    
    # create a directory to dump images
    images_path = os.path.join(dataset_path, 'images')
    if not os.path.exists(images_path):
        os.makedirs(images_path)

    # create tf dataset
    tf_dataset, total_images = get_dataset(
        dataset_path=dataset_path,
        split=args.split,
        batch_size=args.batch_size,
        image_size=args.image_size
    )

    # create json and dump images while creating it
    json_dataset = process_dataset(
        dataset=tf_dataset,
        images_path=images_path,
        total_images=total_images, 
        batch_size=args.batch_size
    )
    
    # dump json
    json_filename = os.path.join(dataset_path, f'{args.split}.json')
    with open(json_filename, 'w') as f:
        json.dump(json_dataset, f)