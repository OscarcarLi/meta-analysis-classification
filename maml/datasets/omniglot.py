import os
import glob
import random
from collections import defaultdict

import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import Omniglot
from torchvision.datasets.utils import list_files

from maml.sampler import ClassBalancedSampler
from maml.datasets.task import Task


class OmniglotMAMLSplit(Omniglot):
    """Implements similar train / test split for Omniglot as
    https://github.com/cbfinn/maml/blob/master/data_generator.py

    Uses torchvision.datasets.Omniglot for downloading and checking
    dataset integrity.
    A map-style dataset where we use index to get the image and class
    """
    def __init__(self, root, split='train', num_train_classes=1100, **kwargs):
        '''
        __init__(self, root, background=True, transform=None, target_transform=None,
                 download=False)
        '''
        super(OmniglotMAMLSplit, self).__init__(root, download=True,
                                                background=True, **kwargs)

        self._split = split
        self._num_train_classes = num_train_classes

        # download testing data and test integrity
        self.background = False
        self.download()
        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted')

        # a list of strings like this
        # 'data/omniglot-py/images_evaluation/Gurmukhi/character17'
        # totally there are 1623 characters
        all_character_dirs = glob.glob(os.path.join(self.root, '**/**/**'))
        if self._split == 'train':
            print('Omniglot train')
            self._character_dirs = all_character_dirs[:self._num_train_classes]
        else:
            print('Omniglot test')
            self._character_dirs = all_character_dirs[self._num_train_classes:]

        self._character_images = []
        for i, char_path in enumerate(self._character_dirs):
            # cp is the path to a specific image file, i is the absolute class index among all the classes in the set
            img_list = [(cp, i) for cp in glob.glob(char_path + '/*')]
            self._character_images.append(img_list)

        self._flat_character_images = sum(self._character_images, [])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the absolute character class.
        """
        image_path, character_class = self._flat_character_images[index]
        # convert('L') convert to gray scale
        # L = R * 299/1000 + G * 587/1000 + B * 114/1000
        image = Image.open(image_path, mode='r').convert('L')

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            character_class = self.target_transform(character_class)

        return image, character_class


class OmniglotMetaDataset(object):
    """
    TODO: Check if the data loader is fast enough.
    Args:
        root: path to omniglot dataset
        img_side_len: images are scaled to this size
        num_classes_per_batch: number of classes to sample for each batch
        num_samples_per_class: number of samples to sample for each class
            for each batch. For K shot learning this should be K + number
            of validation samples
        num_total_batches: total number of tasks to generate
        train: whether to create data loader from the test or validation data
    """
    def __init__(self, name='Omniglot', root='data', 
                 img_side_len=28, img_channel=1,
                 num_classes_per_batch=5, num_samples_per_class=5, 
                 num_total_batches=200000,
                 num_val_samples=1, meta_batch_size=40, split='train',
                 num_train_classes=1100, num_workers=0, device='cpu'):
        self.name = name
        self._root = root
        self._img_side_len = img_side_len
        self._img_channel = img_channel
        self._num_classes_per_batch = num_classes_per_batch
        self._num_train_samples_per_class = num_samples_per_class
        self._num_total_batches = num_total_batches
        self._num_val_samples = num_val_samples
        self._meta_batch_size = meta_batch_size
        self._num_train_classes = num_train_classes
        self._split = split
        self._num_workers = num_workers
        self._device = device

        self._total_samples_per_class = self._num_train_samples_per_class + self._num_val_samples
        self._dataloader = self._get_omniglot_data_loader()

        self.input_size = (img_channel, img_side_len, img_side_len)
        self.output_size = self._num_classes_per_batch

    def _get_omniglot_data_loader(self):
        assert self._img_channel == 1 or self._img_channel == 3
        resize = transforms.Resize(self._img_side_len, Image.LANCZOS)
        invert = transforms.Lambda(lambda x: 1.0 - x)
        if self._img_channel > 1:
            # tile the image
            # number of times to repeat for each dimension (ch, h, w)
            tile = transforms.Lambda(lambda x: x.repeat(self._img_channel, 1, 1))
            img_transform = transforms.Compose(
                [resize, transforms.ToTensor(), invert, tile])
        else:
            img_transform = transforms.Compose(
                [resize, transforms.ToTensor(), invert])
        dset = OmniglotMAMLSplit(self._root, transform=img_transform,
                                 split=self._split,
                                 num_train_classes=self._num_train_classes)
        # repeats in labels [0, 0, ..., 2, 2, ..., ...., 1099]
        _, labels = zip(*dset._flat_character_images)
        sampler = ClassBalancedSampler(dataset_labels=labels,
                                       num_classes_per_batch=self._num_classes_per_batch,
                                       num_samples_per_class=self._total_samples_per_class,
                                       meta_batch_size=self._meta_batch_size,
                                       num_total_batches=self._num_total_batches)

        batch_size = (self._meta_batch_size
                      *self._num_classes_per_batch
                      * self._total_samples_per_class
                      )
        loader = DataLoader(dset, batch_size=batch_size, sampler=sampler,
                            num_workers=self._num_workers, pin_memory=True)
        return loader

    def _make_single_batch(self, img_list, label_list):
        """Split imgs and labels into train and validation set.
        TODO: check if this might become the bottleneck"""
        # label_list is a list of absolute labels for each img in the img_list
        # relabel classes randomly
        new_labels = list(range(self._num_classes_per_batch))
        random.shuffle(new_labels)
        label_list = label_list.tolist()
        label_set = set(label_list)
        # label_map maps absolute label to local relative label
        label_map = {label: i for i, label in zip(new_labels, label_set)}
        # change the label in labels to be within the range _num_classes_per_batch
        updated_label_list = [label_map[l] for l in label_list]

        label_to_indices = defaultdict(list)
        # label_indices maps every new [0, num_classes) label to the list of image indices
        for i, label in enumerate(updated_label_list):
            label_to_indices[label].append(i)

        # rotate randomly to create new classes
        # TODO: move this to torch once supported.
        for label, indices in label_to_indices.items():
            # rotate all the instances of the same class by a random degree
            rotation = np.random.randint(4)
            for idx in indices:
                img = img_list[idx].numpy()
                # copy here for contiguity
                img = np.copy(np.rot90(img, k=rotation, axes=(1,2)))
                img_list[idx] = torch.from_numpy(img)

        # assign samples to train and validation sets
        val_indices = []
        train_indices = []
        for label, indices in label_to_indices.items():
            val_indices.extend(indices[:self._num_val_samples])
            train_indices.extend(indices[self._num_val_samples:])
        label_tensor = torch.tensor(updated_label_list, device=self._device)
        img_list = img_list.to(self._device)
        train_task = Task(img_list[train_indices], label_tensor[train_indices], self.name)
        val_task = Task(img_list[val_indices], label_tensor[val_indices], self.name)

        return train_task, val_task

    def _make_meta_batch(self, imgs, labels):
        batches = []
        inner_batch_size = (self._total_samples_per_class
                            * self._num_classes_per_batch)
        for i in range(0, len(imgs) - 1, inner_batch_size):
            batch_imgs = imgs[i:i+inner_batch_size]
            batch_labels = labels[i:i+inner_batch_size]
            batch = self._make_single_batch(batch_imgs, batch_labels)
            batches.append(batch)

        train_tasks, val_tasks = zip(*batches)

        return train_tasks, val_tasks

    def __iter__(self):
        # the dataloader produces the images to be made into a batch of tasks
        for imgs, labels in iter(self._dataloader):
            # imgs and labels are the images and absolute labels for a meta-batch 
            train_tasks, val_tasks = self._make_meta_batch(imgs, labels)
            yield train_tasks, val_tasks
