from collections import defaultdict
from PIL import Image
import torchvision.transforms as transforms
import random
import torch
import json
import numpy as np
from tqdm import tqdm
from collections import Counter
import concurrent.futures


def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    return img


class FedDataLoader:
    def __init__(
            self,
            dataset,
            batch_size,
            n_batches=0,):
        """Federated dataset's dataloader

        Args:
            dataset (FedDataset): a dataset with __getitem__(client_id) function that returns
                                    (support_x, support_y, query_x, query_y)
            n_batches (int): number of batches of users (aka tasks) the dataloader returns; if batch_size == 0, then give a full pass of all clients in batch_size without repetition.
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
        # self.n_way = self.dataset.n_way
        # self.n_shot = self.dataset.n_shot_per_class
        # self.n_query = self.dataset.n_query_per_class
        # self.randomize_query = self.dataset.randomize_query

    
    def __iter__(self):
        '''
        every time return
                batch_support_x (batch_size, number_of_avaliable_classes * n_shot_per_class, c, h, w)
                batch_support_y (batch_size, number_of_avaliable_classes * n_shot_per_class,)
                batch_query_x (batch_size, number_of_avaliable_classes * n_query_per_class, c, h, w)
                batch_query_y (batch_size, number_of_avaliable_classes * n_query_per_class,)
        '''
        return iter(self.data_loader)
    
    def __len__(self):
        return len(self.data_loader)


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
        if self.n_batches != 0:
            for i in range(self.n_batches):
                # sample without replacement self.batch_size number of clients
                yield random.sample(population=self.client_id_list, k=self.batch_size)
        else:
            # when n_batches == 0, we just loop over the entire dataset with a shuffled client order
            random_client_list = random.sample(population=self.client_id_list, k=len(self.client_id_list))
            i = 0
            while i < len(random_client_list):
                old_i = i
                i = min(i + self.batch_size, len(random_client_list))
                yield random_client_list[old_i: i]


def construct_client_dataset(input):
    # for parallel client loading used in FedDataset_Fix 

    client_id, class_to_sqimagepathlist, kwargs = input
    class_str_list = list(class_to_sqimagepathlist.keys())
    # json.load can only return dictionary with key values as string
    # convert the string to integer for every class
    # this might not be necessary if we are not use cl as the label
    for cl_str in class_str_list:
        class_to_sqimagepathlist[int(cl_str)] = class_to_sqimagepathlist[cl_str]
        del class_to_sqimagepathlist[cl_str]

    return ClientDataset_Fix(
                client_id=client_id,
                class_to_sqimagepathlist=class_to_sqimagepathlist,
                **kwargs)


class FedDataset_Fix(torch.utils.data.Dataset):
    def __init__(self,
                json_path,
                image_size=None,
                preload=False):

        """Dataset object that organizes all user's data through ClientDataset

        Args:
            json_path (str): the path for the json file that has the hierarchy:
                                client_id -> integer_class -> list of image_paths
            # n_shot_per_class (int): number of support examples per class
            # n_query_per_class (int): number of query examples per class
            # n_way (int): the number of ways to sample from.
                        # Defaults to 0 which then will sample from every single class.
            image_size (tuple of ints, optional): reshape the image to image_size.
                                        Defaults to None which means no resizing.
            # randomize_query (bool, optional): whether to have random number of query points for each class
            preload (bool, optional): whether to have every client load the data into memory.
                                        Defaults to False.
            # fixed_sq (bool, optional): if true, for every client in the dataset, always use the same support and query set.
        """

        with open(json_path, 'r') as file:
            client_to_class_to_sqimagepathlist = json.load(file, parse_int=True)

        self.client_dict = {}
        kwargs = {
            'image_size': image_size,
            'preload': preload,
        }
        input_list = [(client_id, class_to_sqimagepathlist, kwargs) \
                        for client_id, class_to_sqimagepathlist in client_to_class_to_sqimagepathlist.items()]


        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        # with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
                # Process the list of files, but split the work across the thread pools!
                for client in tqdm(executor.map(construct_client_dataset, input_list)):
                    self.client_dict[client.client_id] = client

        # self.n_way is a legacy field requirement for algorithm
        # self.n_way = next(iter(self.client_dict.values())).n_way # currently a hacky way to count number of ways from an arbitrary client
        # self.n_shot_per_class = (next(iter(self.client_dict.values()))).n_shot_per_class
        # self.n_query_per_class = (next(iter(self.client_dict.values()))).n_query_per_class

    def __len__(self):
        return len(self.client_dict.keys())

    def __getitem__(self, client_id):
        # given a client_id, return a sampled tuple 
        #             (support_x, support_y, query_x, query_y)
        return self.client_dict[client_id].sample()

    def client_id_list(self):
        return list(sorted(self.client_dict.keys()))


def construct_defaultdictlist():
    return defaultdict(list)


class ClientDataset_Fix:
    def __init__(self,
                 client_id,
                 class_to_sqimagepathlist,
                 image_size=None,
                 preload=False,
                #  fixed_sq=False,
                #  fixed_n_shot=None,
                #  fixed_n_query=None):
                # augmentation or not
                ):
        """A data structure for sampling a specific client's data

        Args:
            client_id (str): the unique identifier of the client
            class_to_sqimagepathlist (dict): maps a class integer to a dictionary
                                           with 'support': support_list
                                           with 'query': query_list
            image_size (tuple of ints, optional): reshape the image to image_size (h,w).
                                        cannot pass a single int because it would only match
                                        the shorter sidelength to the image_size but not the longer side.
                                        Defaults to None which means no resizing.
            preload (bool, optional): whether to load the data into memory. Defaults to False.
        """
        self.client_id = client_id

        self.classes = list(sorted(class_to_sqimagepathlist.keys()))
        self.class_to_sqimagepathlist = class_to_sqimagepathlist

        # resize image to the image_size (can be a single integer or a tuple)
        if image_size is not None:
            self.transform = transforms.Compose(
                [transforms.Resize(size=image_size),
                transforms.ToTensor()])
        else:
            self.transform = transforms.ToTensor()

        self.image_size = image_size
        self.n_way = len(class_to_sqimagepathlist) # this can be inferred
        # print(next(iter(class_to_sqimagepathlist.values())))
        # self.n_shot_per_class = len(next(iter(class_to_sqimagepathlist.values()))['support'])
        # self.n_query_per_class = len(next(iter(class_to_sqimagepathlist.values()))['query'])

        self.preload = preload
        if self.preload:
            self.class_to_sqimagelist = defaultdict(construct_defaultdictlist) # defaultdict(lambda: defaultdict(list))
            for cl, sq_to_imagepathlist in self.class_to_sqimagepathlist.items():
                for image_path in sq_to_imagepathlist['support']:
                    self.class_to_sqimagelist[cl]['support'].append(self.transform(load_image(image_path)))
                for image_path in sq_to_imagepathlist['query']:
                    self.class_to_sqimagelist[cl]['query'].append(self.transform(load_image(image_path)))


    def sample(self):
        """
        For every class of which the client has data,
            sample n_shot_per_class for support set
        and sample n_query_per_class for query set
        if n_way is 0, sample from every class the client has, otherwise sample randomly
            only n_way classes
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

        # will sample every class; no label shuffling
        classes_to_sample = self.classes
        # random.shuffle(classes_to_sample)

        for i, cl in enumerate(classes_to_sample):
            if self.preload:
                cl_support_examples = self.class_to_sqimagelist[cl]['support']
                cl_query_examples = self.class_to_sqimagelist[cl]['query']

            else:
                support_paths = self.class_to_sqimagepathlist[cl]['support']
                cl_support_examples = [self.transform(load_image(path)) for path in support_paths]

                query_paths = self.class_to_sqimagepathlist[cl]['query']
                cl_query_examples = [self.transform(load_image(path)) for path in query_paths]

            support_x.extend(cl_support_examples)
            support_y.extend([i] * len(cl_support_examples))
            query_x.extend(cl_query_examples)
            query_y.extend([i] * len(cl_query_examples))

        support_x = torch.stack(support_x, dim=0)
        support_y = torch.tensor(support_y)
        query_x = torch.stack(query_x, dim=0)
        query_y = torch.tensor(query_y)

        return (support_x, support_y, query_x, query_y)



class SimpleFedDataset(torch.utils.data.Dataset):

    def __init__(self, json_path,
                       preload,
                       image_size,
                       verbose=True):

        """[summary]

        Args:
            json_path (str): the path for the json file that has the hierarchy:
                                client_id -> integer_class -> list of image_paths
            preload (bool, optional): whether to load the data into memory. Defaults to False.
            image_size (int): the side length of the square image
            verbose (bool, optional): log dataset creation details. Defaults to True.
        """

        # load json
        with open(json_path, 'r') as file:
            client_to_class_to_imagepathlist = json.load(file, parse_int=True)

        # initialize lists
        self.imagepaths = [] # list of all image paths
        self.labels = [] # list of all labels

        # iterate over json elements
        for client_id, class_to_imagepathlist in tqdm(client_to_class_to_imagepathlist.items()):

            # json.load can only return dictionary with key values as string
            # convert the string to integer for every class
            # then append the corresponding labels and imagepathlist to global vars self.labels and self.imagepaths
            for cl_str, imagepathlist in class_to_imagepathlist.items():
                self.imagepaths += imagepathlist
                self.labels += [int(cl_str)] * len(imagepathlist)

        # transforms
        # resize image to the image_size (can be a single integer or a tuple)
        self.image_size = image_size
        if self.image_size is not None:
            self.transform = transforms.Compose(
                [transforms.Resize(size=self.image_size),
                transforms.ToTensor()])
        else:
            self.transform = transforms.ToTensor()

        # preloading
        self.preload = preload
        if self.preload:
            print("Please wait ... preloading images.")
            self.images = []
            with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
                # Process the list of files, but split the work across the process pool to use all CPUs!
                for image in tqdm(executor.map(load_image, self.imagepaths)):
                    self.images.append(image)
            print(f"Preloading done. Have {len(self.images)} images loaded in memory.")
        else:
            self.images = self.imagepaths
            print(f"No preloading done. Have {len(self.images)} imagepaths.")

        
        # logs
        if verbose:
            print(f"No. of classes in dataset {set(self.labels)}")
            print(f"No. of images per class in the dataset {Counter(self.labels)}")
            


    def __getitem__(self, i):
        if self.preload:
            img = self.images[i]
        else:
            img = load_image(self.images[i])
        transformed_img = self.transform(img)
        return transformed_img, self.labels[i]


    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    train_meta_dataset = FedDataset_Fix(
                        json_path='../../fed_data/celeba/base.json',
                        n_shot_per_class=1,
                        n_query_per_class=5,
                        image_size=(84, 84), # has to be a (h, w) tuple
                        randomize_query=False,
                        preload=True,
                        fixed_sq=False)

    train_loader = FedDataLoader(
                        dataset=train_meta_dataset,
                        n_batches=200,
                        batch_size=20)