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
This would need additional params: [n_way, n_shot, n_query, n_eposide]
"""
class MetaDataManager(DataManager):
    
    def __init__(self, dataset, batch_size, n_batches, n_way):        
        """object to create the dataloader

        Args:
            dataset (MetaDataset): given a class index, return random support and query examples
            batch_size (int): the number of tasks in a task batch
            n_batches (int): total number of task batches
            n_way (int): number of ways in a task
        """        
        super(MetaDataManager, self).__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.n_way = n_way
        self.n_shot = dataset.n_shot
        self.n_query = dataset.n_query
        self.sampler = EpisodicBatchSampler(
                            n_classes=len(dataset),
                            n_way=self.n_way, 
                            n_batches=self.n_batches,
                            n_tasks=self.batch_size)  
        
    def get_data_loader(self):
        data_loader_params = dict(batch_sampler=self.sampler,
                                  num_workers=12,
                                  pin_memory=True)

        data_loader = torch.utils.data.DataLoader(self.dataset, **data_loader_params)
        return data_loader



"""
Samples n_way random classes for each episode.
EpisodicBatchSampler is hereditary. 
Used by almost all pytorch implementations released after Protonet.
"""

class EpisodicBatchSampler(torch.utils.data.Sampler):

    def __init__(self, n_classes, n_way, n_tasks, n_batches):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_tasks = n_tasks
        self.n_batches = n_batches

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        for i in range(self.n_batches):
            '''
            for self.n_batches number of times,
            each time return the sampled classes' indices for self.n_tasks
            '''
            yield np.concatenate(
                [np.random.choice(self.n_classes, self.n_way, replace=False) for _ in range(self.n_tasks)])
