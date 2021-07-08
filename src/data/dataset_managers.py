import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from abc import abstractmethod
import torch
from PIL import ImageEnhance

from collections import defaultdict
import data

from src.data.datasets import MultipleMetaDatasets


"""
Data Manager for meta-training methods.
This would need additional params: [n_way, n_shot, n_query, n_eposide]
"""
class MetaDataLoader:
    
    def __init__(self,
                dataset,
                n_batches,
                batch_size,
                n_way=None,
                n_shot=None,
                n_query=None,
                randomize_query=False,
                verbose=True,
                p_dict=None):        
        """object to create the dataloader
        Args:
            dataset (MetaDataset): given a class index, return random support and query examples
            batch_size (int): the number of tasks in a task batch
            n_batches (int): total number of task batches
            n_way (int): number of ways in a task
            n_shot (int): number of support examples per class
            n_query (int): average number of query examples per class
            randomize_query (bool): whether to use exactly the same number of examples per class.
            p_dict (dict): maps a class to its probability of being selected, defaults to None (uniform prob.)
        """        
        # super(MetaDataLoader, self).__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.randomize_query = randomize_query
        self.p_dict = p_dict

        print("Size of Support:", self.n_shot)
        print("Size of Query:", self.n_query, "randomize query", self.randomize_query)
        print("p_dict", self.p_dict)
        
        if isinstance(dataset, MultipleMetaDatasets):
            self.sampler = MultipleMetaDatasetsEpisodicBatchSampler(
                    multi_dataset=dataset,
                    n_way=self.n_way, 
                    n_shot=self.n_shot,
                    n_query=self.n_query,
                    random_query=self.randomize_query,
                    n_batches=self.n_batches,
                    n_tasks=self.batch_size,
                    p_dict=p_dict,
                    verbose=verbose)
        else:        
            self.sampler = EpisodicBatchSampler(
                    dataset=dataset,
                    n_way=self.n_way, 
                    n_shot=self.n_shot,
                    n_query=self.n_query,
                    random_query=self.randomize_query,
                    n_batches=self.n_batches,
                    n_tasks=self.batch_size,
                    p_dict=p_dict,
                    verbose=verbose)

        self.data_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_sampler=self.sampler,
            num_workers=24,
            pin_memory=False,
            collate_fn=lambda ls: collate_fn(
                                    ls=ls,
                                    has_support=(self.n_shot != 0),
                                    has_query=(self.n_query != 0)),
            # persistent_workers=False, # persistent_workers have conflict with pin_memory
        )

    def __iter__(self):
        return iter(self.data_loader)
    
    def __len__(self):
        return len(self.data_loader)



"""
Samples n_way random classes for each episode.
EpisodicBatchSampler is hereditary. 
Used by almost all pytorch implementations released after Protonet.
"""

class EpisodicBatchSampler(torch.utils.data.Sampler):

    def __init__(self, dataset, n_way, n_shot, n_query, random_query, n_tasks, n_batches, p_dict, verbose=True):
        self.dataset = dataset
        self.classes = self.dataset.classes
        self.n_classes = len(self.classes)
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.random_query = random_query
        self.n_tasks = n_tasks
        self.n_batches = n_batches

        # construct array of probabilities for sampler
        if p_dict is None:
            self.p = np.ones(self.n_classes) / self.n_classes
        else:
            self.p = []
            for cl in self.classes:
                self.p.append(p_dict[cl])
        if verbose:
            print("Setting an episodic sampler over classes")
            for cl, prob in zip(self.classes, self.p):
                print(f'({cl}, {prob})')

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        for _ in range(self.n_batches):
            '''
            for self.n_batches number of times,
            each time return the sampled classes' indices for self.n_tasks
            '''
            yield_result = []
            for task_idx in range(self.n_tasks):
                # first determine how many samples to create for each class
                if self.random_query:
                    # here add np.ones ensure every class has at least one query example
                    counts = np.random.multinomial(
                                n=(self.n_query - 1) * self.n_way,
                                pvals=[1/self.n_way] * self.n_way) + \
                                np.ones(shape=self.n_way, dtype=int)
                else:
                    counts = [self.n_query] * self.n_way

                # choose unique classes (the class composition for this task)
                classes_chosen = np.random.choice(
                                    self.classes,
                                    self.n_way,
                                    replace=False,
                                    p=self.p)

                # label these classes with possible label shuffling.
                cl_labels = np.random.permutation(self.n_way)

                for cl, n_query, cl_label in zip(classes_chosen, counts, cl_labels):
                    yield_result.append({
                        'task_idx': task_idx,
                        'cl': cl,
                        'n_shot': self.n_shot,
                        'n_query': n_query,
                        'cl_label': cl_label,
                    })

            yield yield_result




class MultipleMetaDatasetsEpisodicBatchSampler(torch.utils.data.Sampler):

    def __init__(self, multi_dataset, n_way, n_shot, n_query, random_query, n_tasks, n_batches, p_dict, verbose=True):
        
        self.multi_dataset = multi_dataset
        self.samplers = {}
        self.n_batches = n_batches
        self.n_tasks = n_tasks
        assert random_query == False
        assert p_dict is None
        print("n_way", n_way, n_shot, n_query)
        for dataset_name in self.multi_dataset.datasets:
            self.samplers[dataset_name] = EpisodicBatchSampler(
                dataset=self.multi_dataset.datasets[dataset_name],
                n_way=n_way, 
                n_shot=n_shot,
                n_query=n_query,
                random_query=random_query,
                n_batches=1,
                n_tasks=1,
                p_dict=p_dict,
                verbose=verbose
            )

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        for _ in range(self.n_batches):
            '''
            for self.n_batches number of times,
            each time return the sampled classes' indices for self.n_tasks
            '''
            yield_result = []
            chosen_datasets = np.random.choice(
                    list(self.multi_dataset.datasets.keys()), self.n_tasks, replace=True)
            for task_idx, chosen_dataset_name in enumerate(chosen_datasets):
                chosen_dataset_yield = next(iter(self.samplers[chosen_dataset_name]))
                for class_yield in chosen_dataset_yield:
                    class_yield['dataset_idx'] = chosen_dataset_name
                    class_yield['task_idx'] = task_idx
                yield_result += chosen_dataset_yield
            yield yield_result
                


class GoogleMetaDatasetEpisodicBatchSampler:

    def __init__(self, n_tasks, n_batches, n_datasets):
        self.n_tasks = n_tasks 
        self.n_batches = n_batches
        self.n_datasets = n_datasets

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        for _ in range(self.n_batches):
            '''
            for self.n_batches number of times,
            each time return the self.n_tasks episodes from GoogleMetaDataset
            '''
            yield_result = np.random.choice(self.n_datasets, self.n_tasks, replace=False)
            print(yield_result)
            yield yield_result
    
    




def collate_fn(ls, has_support, has_query):
    result = defaultdict(lambda: defaultdict(list))

    for task_cl_dict in ls:
        task_idx = task_cl_dict['task_idx']
        if has_support:
            result[task_idx]['support_x'].append(task_cl_dict['support_x_cl'])
            result[task_idx]['support_y'].append(task_cl_dict['support_y_cl'])
        if has_query:
            result[task_idx]['query_x'].append(task_cl_dict['query_x_cl'])
            result[task_idx]['query_y'].append(task_cl_dict['query_y_cl'])

    task_indices = sorted(result.keys())

    if has_support:
        support_x_tb = torch.stack(
            [torch.cat(result[task_idx]['support_x'], dim=0)
                for task_idx in task_indices],
            dim=0)
        support_y_tb = torch.stack(
            [torch.cat(result[task_idx]['support_y'], dim=0)
                for task_idx in task_indices],
            dim=0)
    
    if has_query:
        query_x_tb = torch.stack(
            [torch.cat(result[task_idx]['query_x'], dim=0)
                for task_idx in task_indices],
            dim=0)
        query_y_tb = torch.stack(
            [torch.cat(result[task_idx]['query_y'], dim=0)
                for task_idx in task_indices],
            dim=0)

    if has_support and has_query:
        return (support_x_tb, support_y_tb, query_x_tb, query_y_tb)
    elif has_support and (not has_query):
        return support_x_tb, support_y_tb
    elif (not has_support) and has_query:
        return query_x_tb, query_y_tb
    else:
        assert False, 'no support and no query'