from collections import defaultdict
from PIL import Image
import torchvision.transforms as transforms
import random
import torch
import json
import numpy as np
from tqdm import tqdm


def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    return img


class FedDataLoader:
    def __init__(
            self,
            dataset,
            n_batches,
            batch_size):
        """Federated dataset's dataloader

        Args:
            dataset (FedDataset): a dataset with __getitem__(client_id) function that returns
                                    (support_x, support_y, query_x, query_y)
            n_batches (int): number of batches of users (aka tasks) the dataloader returns
            batch_size (int):  how many users/tasks are in a returned batch
        """        
        
        self.dataset = dataset
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.batch_sampler = FedBatchSampler(
                                fed_dataset=self.dataset,
                                n_batches=self.n_batches,
                                batch_size=self.batch_size)
        self.data_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_sampler=self.batch_sampler,
            num_workers=8,
            pin_memory=True,
        )

        # these variables are used by algorithm_trainer.py but these can actually be inferred from support_x, support_y
        self.n_way = self.dataset.n_way
        self.n_shot = self.dataset.n_shot_per_class
        self.n_query = self.dataset.n_query_per_class
        self.randomize_query = self.dataset.randomize_query

    
    def __iter__(self):
        '''
        every time return
                batch_support_x (batch_size, number_of_avaliable_classes * n_shot_per_class, c, h, w)
                batch_support_y (batch_size, number_of_avaliable_classes * n_shot_per_class,)
                batch_query_x (batch_size, number_of_avaliable_classes * n_query_per_class, c, h, w)
                batch_query_y (batch_size, number_of_avaliable_classes * n_query_per_class,)
        '''
        return iter(self.data_loader)


class FedBatchSampler(torch.utils.data.Sampler):
    def __init__(
            self,
            fed_dataset,
            n_batches,
            batch_size):
        self.fed_dataset = fed_dataset
        self.client_id_list = self.fed_dataset.client_id_list()
        self.n_batches = n_batches
        self.batch_size = batch_size


    def __len__(self):
        return self.n_batches
    
    def __iter__(self):
        for i in range(self.n_batches):
            # sample without replacement self.batch_size number of clients
            yield random.sample(population=self.client_id_list, k=self.batch_size)


class FedDataset(torch.utils.data.Dataset):
    def __init__(self,
                json_path,
                n_shot_per_class,
                n_query_per_class,
                image_size=None,
                randomize_query=False,
                preload=False):
        """Dataset object that organizes all user's data through ClientDataset

        Args:
            json_path (str): the path for the json file that has the hierarchy:
                                client_id -> integer_class -> list of image_paths
            n_shot_per_class (int): number of support examples per class
            n_query_per_class (int): number of query examples per class
            image_size (int or tuple of ints, optional): reshape the image to image_size.
                                        Defaults to None which means no resizing.
            randomize_query (bool): whether to have random number of query points for each class
            preload (bool, optional): whether to have every client load the data into memory.
                                        Defaults to False.
        """                

        with open(json_path, 'r') as file:
            client_to_class_to_imagepathlist = json.load(file, parse_int=True)

        self.client_dict = {}
        for client_id, class_to_imagepathlist in tqdm(client_to_class_to_imagepathlist.items()):

            # json.load can only return dictionary with key values as string
            # convert the string to integer for every class
            class_str_list = list(class_to_imagepathlist.keys())
            self.n_way = len(class_str_list) # currently a hacky way to do this
            for cl_str in class_str_list:
                class_to_imagepathlist[int(cl_str)] = class_to_imagepathlist[cl_str]
                del class_to_imagepathlist[cl_str]

            self.client_dict[client_id] = \
                ClientDataset(
                        client_id=client_id,
                        class_to_imagepathlist=class_to_imagepathlist,
                        image_size=image_size,
                        preload=preload)
        
        self.n_shot_per_class = n_shot_per_class
        self.n_query_per_class = n_query_per_class
        # currently no implementation for randomize_query but it would happen in this ClientDataset
        # object in sample()
        self.randomize_query = randomize_query

    def __len__(self):
        return len(self.client_dict.keys())

    def __getitem__(self, client_id):
        # given a client_id, return a sampled tuple 
        #             (support_x, support_y, query_x, query_y)
        return self.client_dict[client_id].sample(
                    n_shot_per_class=self.n_shot_per_class,
                    n_query_per_class=self.n_query_per_class,
                    randomize_query=self.randomize_query)
    
    def client_id_list(self):
        return list(sorted(self.client_dict.keys()))


class ClientDataset:
    def __init__(self,
                 client_id,
                 class_to_imagepathlist,
                 image_size=None,
                 preload=False):
                # augmentation or not
        """A data structure for sampling a specific client's data

        Args:
            client_id (str): the unique identifier of the client
            class_to_imagepathlist (dict): maps a class integer to the list of image_paths
            image_size (tuple of ints, optional): reshape the image to image_size (h,w).
                                        cannot pass a sinlge int because it would only match
                                        the shorter sidelength to the image_size but not the longer side.
                                        Defaults to None which means no resizing.
            preload (bool, optional): whether to load the data into memory. Defaults to False.
        """
        self.client_id = client_id

        self.classes = list(sorted(class_to_imagepathlist.keys()))
        self.class_to_imagepathlist = class_to_imagepathlist

        # resize image to the image_size (can be a single integer or a tuple)
        if image_size is not None:
            self.transform = transforms.Compose(
                [transforms.Resize(size=image_size),
                transforms.ToTensor()])
        else:
            self.transform = transforms.ToTensor()

        self.image_size = image_size

        self.preload = preload
        if self.preload:
            self.class_to_imagelist = defaultdict(list)
            for cl, imagepathlist in self.class_to_imagepathlist.items():
                for image_path in imagepathlist:
                    self.class_to_imagelist[cl].append(self.transform(load_image(image_path)))


    def sample(self, n_shot_per_class, n_query_per_class, randomize_query=False):
        """
        For every class of which the client has data,
            sample n_shot_per_class for support set
        and sample n_query_per_class for query set
        Returns:
            (support_x, support_y, query_x, query_y):
                support_x (number_of_avaliable_classes * n_shot_per_class, c, h, w)
                support_y (number_of_avaliable_classes * n_shot_per_class,)
                query_x (number_of_avaliable_classes * n_query_per_class, c, h, w)
                query_y (number_of_avaliable_classes * n_query_per_class,)
        """        
        support_x = []
        support_y = []

        query_x = []
        query_y = []

        num_classes = len(self.classes)
        if randomize_query:
            # randomize query so that not every class will have the same number
            # n_query_per_class, but the total number of query points is still equal to 
            # n_query_per_class * num_classes
            num_queries = np.random.multinomial(
                                n=(n_query_per_class - 1) * num_classes,
                                pvals=[1/num_classes] * num_classes) + \
                                np.ones(shape=num_classes, dtype=int)
        else:
            num_queries = [n_query_per_class] * num_classes

        for cl, n_q in zip(self.classes, num_queries):
            if self.preload:
                # Return a k sized list of elements chosen from the population with replacement.
                examples = random.choices(population=self.class_to_imagelist[cl],
                                k=n_shot_per_class + n_q)
            else:
                example_paths = random.choices(population=self.class_to_imagepathlist[cl],
                                                k=n_shot_per_class + n_q)
                examples = [self.transform(load_image(path)) for path in example_paths]

            support_x.extend(examples[:n_shot_per_class])
            query_x.extend(examples[n_shot_per_class:])
            support_y.extend([cl] * n_shot_per_class)
            query_y.extend([cl] * n_q)
        
        support_x = torch.stack(support_x, dim=0)
        support_y = torch.tensor(support_y)
        query_x = torch.stack(query_x, dim=0)
        query_y = torch.tensor(query_y)

        return (support_x, support_y, query_x, query_y)