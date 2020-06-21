import numpy as np
from itertools import chain
from maml.datasets.task import Task


class MultimodalFewShotDataset(object):

    def __init__(self, datasets, num_total_batches,
                 name='MultimodalFewShot',
                 mix_meta_batch=True, mix_mini_batch=False,
                 split='train', verbose=False, txt_file=None):
        # datasets is a list of datasets
        self._datasets = datasets
        self._num_total_batches = num_total_batches
        self.name = name
        self.num_dataset = len(datasets)
        self.dataset_names = [dataset.name for dataset in self._datasets]
        self._meta_batch_size = datasets[0]._meta_batch_size
        # mix_meta_batch means whether the meta task batch will have tasks from different task classes
        self._mix_meta_batch = mix_meta_batch
        # mix_mini_batch means if we are mixing meta batch, do you do this completely randomly or follow a fixed order
        self._mix_mini_batch = mix_mini_batch
        self._split = split
        if self._split == 'test':
            print('multimodal test sampling')
        elif self._split == 'val':
            print('multimodal val sampling')
        else :
            print('multimodal train sampling')
        self._verbose = verbose
        self._txt_file = open(txt_file, 'w') if not txt_file is None else None
        
        # make sure all input/output sizes match
        input_size_list = [dataset.input_size for dataset in self._datasets]
        assert input_size_list.count(input_size_list[0]) == len(input_size_list)
        output_size_list = [dataset.output_size for dataset in self._datasets]
        assert output_size_list.count(output_size_list[0]) == len(output_size_list)
        self.input_size = datasets[0].input_size
        self.output_size = datasets[0].output_size

        # build iterators
        self._datasets_iter = [iter(dataset) for dataset in self._datasets]
        if self._split != 'train':
            self._iter_index = 0
        
        self.n = 0 # the number of batches produced so far

        # print info
        print('Multimodal Few Shot Datasets: {}'.format(' '.join(self.dataset_names)))
        print('mix meta batch: {}'.format(mix_meta_batch))
        print('mix mini batch: {}'.format(mix_mini_batch))

    def __next__(self):
        if self.n < self._num_total_batches:
            self.n += 1
            # mix_meta_batch defaults to True
            # mix_mini_batch defaults to False
            if not self._mix_meta_batch and not self._mix_mini_batch:
                # sample all of the tasks from one of the task classes
                dataset_index = np.random.randint(len(self._datasets))
                if self._verbose:
                    print('Sample from: {}'.format(self._datasets[dataset_index].name))
                train_tasks, val_tasks = next(self._datasets_iter[dataset_index])
                return train_tasks, val_tasks
            else: 
                # get all tasks
                tasks = []
                all_train_tasks = []
                all_val_tasks = []
                # one **batch** of tasks from every task class
                for dataset_iter in self._datasets_iter:
                    train_tasks, val_tasks = next(dataset_iter)
                    all_train_tasks.extend(train_tasks)
                    all_val_tasks.extend(val_tasks)
                
                if not self._mix_mini_batch:
                    # mix them to obtain a meta batch
                    """
                    # randomly sample task
                    dataset_indexes = np.random.choice(
                        len(all_train_tasks), size=self._meta_batch_size, replace=False)
                    """
                    # balancedly sample from all datasets
                    dataset_indexes = []
                    if self._split == 'train':
                        dataset_start_idx = np.random.randint(0, self.num_dataset)
                    else:
                        dataset_start_idx = self._iter_index

                    for i in range(self._meta_batch_size):
                        dataset_indexes.append(
                            np.random.randint(0, self._meta_batch_size)+
                            ((i+dataset_start_idx)%self.num_dataset)*self._meta_batch_size)
                    # remember which dataset class was sampled from the last time
                    # if the dataset_start_idx = 3 and num_dataset = 5
                    # then the dataset class we sample from is 3, 4, 0, 1 if our meta_batch_size is 4
                    if self._split != 'train':
                        self._iter_index = (self._iter_index + self._meta_batch_size) % self.num_dataset

                    train_tasks = []
                    val_tasks = []
                    dataset_names = []
                    for dataset_index in dataset_indexes:
                        train_tasks.append(all_train_tasks[dataset_index])
                        val_tasks.append(all_val_tasks[dataset_index])
                        dataset_names.append(self._datasets[dataset_index//self._meta_batch_size].name)
                    if self._verbose:
                        print('Sample from: {} (indexes: {})'.format(
                            [name for name in dataset_names], dataset_indexes))
                    if self._txt_file is not None:
                        for name in dataset_names:
                            self._txt_file.write(name+'\n')
                    return train_tasks, val_tasks
                else:
                    # mix them to obtain a mini batch and make a meta batch
                    raise NotImplementedError
            


        else:
            raise StopIteration

    def __iter__(self):
        # rebuild iterators
        self.n = 0
        self._datasets_iter = [iter(dataset) for dataset in self._datasets]
        return self
