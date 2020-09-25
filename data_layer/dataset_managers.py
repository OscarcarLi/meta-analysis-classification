import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from data_layer.datasets import ClassicalDataset, MetaDataset
from abc import abstractmethod
import torch
from PIL import ImageEnhance



"""
Data augmentation scheme.
aug is True/False acc. in get_composed_transform function.
"""
class TransformLoader:
    def __init__(self, image_size, 
                 normalize_param=dict(mean=[0.485, 0.456, 0.406] , std=[0.229, 0.224, 0.225]),
                 jitter_param=dict(Brightness=0.4, Contrast=0.4, Color=0.4)):
        self.image_size = image_size
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param
    
    def parse_transform(self, transform_type):
        if transform_type=='ImageJitter':
            method = ImageJitter( self.jitter_param )
            return method
        method = getattr(transforms, transform_type)
        if transform_type=='RandomResizedCrop':
            return method(self.image_size) 
        elif transform_type=='CenterCrop':
            return method(self.image_size) 
        elif transform_type=='Resize':
            return method([int(self.image_size*1.15), int(self.image_size*1.15)])
        elif transform_type=='Normalize':
            return method(**self.normalize_param )
        else:
            return method()

    def get_composed_transform(self, dataset, aug = False):
        
        if dataset in ['cifar', 'fc100']:
            mean_pix = [x/255.0 for x in [129.37731888, 124.10583864, 112.47758569]]
            std_pix = [x/255.0 for x in [68.20947949, 65.43124043, 70.45866994]]
            normalize = transforms.Normalize(mean=mean_pix, std=std_pix)
            if aug:
                print("Using meta-optnet version of augmentation")
                transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    transforms.RandomHorizontalFlip(),
                    lambda x: np.array(x),
                    transforms.ToTensor(),
                    normalize
                ])
            else:
                transform = transforms.Compose([
                    lambda x: np.array(x),
                    transforms.ToTensor(),
                    normalize
                ])
        else:
            if aug:
                print("Using our version of augmentation")
                transform_list = ['RandomResizedCrop', 'ImageJitter', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
            else:
                transform_list = ['Resize','CenterCrop', 'ToTensor', 'Normalize']
            transform_funcs = [self.parse_transform(x) for x in transform_list]
            transform = transforms.Compose(transform_funcs)


        return transform




"""
Jitter transform: Brightness, Contrast, Color, Sharpness

Copyright 2017-present, Facebook, Inc.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
transformtypedict=dict(
    Brightness=ImageEnhance.Brightness, 
    Contrast=ImageEnhance.Contrast, 
    Sharpness=ImageEnhance.Sharpness, 
    Color=ImageEnhance.Color
    )

class ImageJitter(object):
    def __init__(self, transformdict):
        self.transforms = [(transformtypedict[k], transformdict[k]) for k in transformdict]


    def __call__(self, img):
        out = img
        randtensor = torch.rand(len(self.transforms))

        for i, (transformer, alpha) in enumerate(self.transforms):
            r = alpha*(randtensor[i]*2.0 -1.0) + 1
            out = transformer(out).enhance(r).convert('RGB')

        return out


"""
Abstract Data Manager class.
"""
class DataManager:
   
    @abstractmethod
    def get_data_loader(self, data_file, aug):
        pass 



"""
Data Manager for classical training methods.
"""
class ClassicalDataManager(DataManager):
    def __init__(self, image_size, batch_size):        
        super(ClassicalDataManager, self).__init__()
        self.batch_size = batch_size
        self.trans_loader = TransformLoader(image_size)

    def get_data_loader(self, dataset, data_file, aug): #parameters that would change on train/val set
        """
        data_file: path to dataset
        aug: boolean to set data augmentation
        """
        transform = self.trans_loader.get_composed_transform(dataset, aug)
        dataset = ClassicalDataset(data_file, transform)
        data_loader_params = dict(batch_size=self.batch_size, shuffle = True, num_workers = 12, pin_memory = True)       
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)

        return data_loader



"""
Data Manager for meta-training methods.
This would beed additional params: [n_way, n_shot, n_query, n_eposide]
"""

class MetaDataManager(DataManager):
    def __init__(self, image_size, n_way, n_shot, n_query, batch_size, fix_support=0, n_episodes=100):        
        super(MetaDataManager, self).__init__()
        self.image_size = image_size
        self.n_way = n_way
        self.batch_size = batch_size
        self.n_shot = n_shot
        self.n_query = n_query
        self.n_episodes = n_episodes
        self.trans_loader = TransformLoader(image_size)
        self.fix_support = fix_support

    def get_data_loader(self, dataset, data_file, support_aug, query_aug): #parameters that would change on train/val set
        """
        data_file: path to dataset
        aug: boolean to set data augmentation
        """
        print("support aug:", support_aug, "query aug:", query_aug)
        support_transform = self.trans_loader.get_composed_transform(dataset, support_aug)
        query_transform = self.trans_loader.get_composed_transform(dataset, query_aug)
        dataset = MetaDataset(data_file, n_shot=self.n_shot, n_query=self.n_query, 
            support_transform=support_transform, query_transform=query_transform, fix_support=self.fix_support)
        sampler = EpisodicBatchSampler(len(dataset), n_way=self.n_way, 
            n_episodes=self.n_episodes, n_tasks=self.batch_size)  
        data_loader_params = dict(batch_sampler=sampler, num_workers=20, pin_memory=True)       
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader




"""
Samples n_way random classes for each episode.
EpisodicBatchSampler is heriditary. 
Used by almost all pytorch implementations released after Protonet.
"""

class EpisodicBatchSampler(object):

    def __init__(self, n_classes, n_way, n_tasks, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_tasks = n_tasks
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield np.concatenate(
                [np.random.choice(self.n_classes, self.n_way, replace=False) for _ in range(self.n_tasks)])
            
