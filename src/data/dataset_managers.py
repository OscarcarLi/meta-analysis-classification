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
    
    def __init__(self, dataset, batch_size, n_batches, n_way, p_dict=None):        
        """object to create the dataloader

        Args:
            dataset (MetaDataset): given a class index, return random support and query examples
            batch_size (int): the number of tasks in a task batch
            n_batches (int): total number of task batches
            n_way (int): number of ways in a task
            p_dict (dict): maps a class to its probability of being selected, defaults to None (uniform prob.)
        """        
        super(MetaDataManager, self).__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.n_way = n_way
        self.n_shot = dataset.n_shot
        self.n_query = dataset.n_query
        self.sampler = EpisodicBatchSampler(
                            classes=dataset.classes,
                            n_way=self.n_way, 
                            n_batches=self.n_batches,
                            n_tasks=self.batch_size,
                            p_dict=p_dict)  
        
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
p_dict is a dictionary that maps a class to its probability of being selected. 
"""

class EpisodicBatchSampler(torch.utils.data.Sampler):

    def __init__(self, classes, n_way, n_tasks, n_batches, p_dict=None):
        self.classes = classes
        self.n_classes = len(classes)
        self.n_way = n_way
        self.n_tasks = n_tasks
        self.n_batches = n_batches
        
        # construct array of probabilities for sampler
        if p_dict is None:
            self.p = np.ones(self.n_classes) / self.n_classes
        else:
            self.p = []
            for cl in self.classes:
                self.p = self.p + [p_dict[cl]]

        print(f"Setting an episodic sampler over classes {list(zip(self.classes, self.p))} ")

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        for i in range(self.n_batches):
            '''
            for self.n_batches number of times,
            each time return the sampled classes' indices for self.n_tasks
            '''
            yield np.concatenate(
                [np.random.choice(self.classes, self.n_way, replace=False, p=self.p) for _ in range(self.n_tasks)])