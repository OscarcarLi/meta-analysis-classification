import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from abc import abstractmethod
import torch
from PIL import ImageEnhance



"""
Abstract Data Manager class.
"""
class DataManager:
   
    @abstractmethod
    def get_data_loader(self, data_file, aug):
        pass 



"""
Data Manager for meta-training methods.
This would beed additional params: [n_way, n_shot, n_query, n_eposide]
"""

class MetaDataManager(DataManager):
    
    def __init__(self, dataset, batch_size, n_episodes, n_way):        
        super(MetaDataManager, self).__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_episodes = n_episodes
        self.n_way = n_way
        self.n_shot = dataset.n_shot
        self.n_query = dataset.n_query
        self.sampler = EpisodicBatchSampler(len(dataset), n_way=self.n_way, 
            n_episodes=self.n_episodes, n_tasks=self.batch_size)  
        
    def get_data_loader(self):
        data_loader_params = dict(batch_sampler=self.sampler, num_workers=12, pin_memory=True)       
        data_loader = torch.utils.data.DataLoader(self.dataset, **data_loader_params)
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
            
