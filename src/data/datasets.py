from h5py._hl import dataset
import torch
from PIL import Image
import json
import numpy as np
import torchvision.transforms as transforms
import concurrent.futures
from collections import defaultdict
import tqdm
import os
import h5py
from copy import deepcopy

from src.data.transforms import TransformLoader

# for metadataset
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
# so that actual tensors are loaded
# when __getitem__ is called
import gin
from meta_dataset.data import config
from meta_dataset.data import dataset_spec as dataset_spec_lib
from meta_dataset.data import learning_spec
from meta_dataset.data import pipeline

GIN_FILE_PATH = 'meta_dataset/learn/gin/setups/data_config.gin'
gin.parse_config_file(GIN_FILE_PATH)
# the above parses default dataset configs for GoogleMetaDataset
# like data augmentation, which is False by default


# for transform
identity = lambda x:x

def load_image(image_path):
    if '.h5' in image_path:
        # image is stored in hdf5 format
        hf = h5py.File(image_path, 'r')
        img_array = hf.get('image')[()]
        img = Image.fromarray(img_array)
    else:
        img = Image.open(image_path).convert('RGB')
    return img



"""
Data Manager for meta-training methods.
This would need additional params: [n_way, n_shot, n_query, n_eposide]
"""


class GoogleMetaDataset:

    def __init__(self, basepath, split, n_ways, n_shots, n_query):
        
        self.all_datasets = [
            'aircraft',
            'cu_birds', 
            'dtd', 
            'fungi', 
            # 'ilsvrc_2012',
            'omniglot', 
            # 'quickdraw', 
            'vgg_flower'
        ]
        
        # paths
        self.basepath = basepath
        
        # ways, shots and query
        self.n_ways = n_ways
        self.n_shots = n_shots
        self.n_query = n_query

        # config specification
        self.config = config.EpisodeDescriptionConfig(
            num_ways=self.n_ways, 
            num_support=self.n_shots, 
            num_query=self.n_query)

        # metadataset specifics
        self.use_bilevel_ontology = False
        self.use_dag_ontology = False

        # split (train/val/test)
        self.split = split
        self.learning_spec_map = {
            'train':learning_spec.Split.TRAIN, 
            'val':learning_spec.Split.VALID, 
            'test':learning_spec.Split.TEST
        }
        self.learning_spec_split = self.learning_spec_map[self.split]
        
        # construct multiple one source batch pipeline for each dataset
        # current implementation can only handle one source per sampler when
        # no of ways, shots, queries are fixed


        # variable_ways_shots = config.EpisodeDescriptionConfig(
        #     num_query=None, num_support=None, num_ways=None)

        # dataset_episodic = pipeline.make_multisource_episode_pipeline(
        #     dataset_spec_list=all_dataset_specs,
        #     use_dag_ontology_list=use_dag_ontology_list,
        #     use_bilevel_ontology_list=use_bilevel_ontology_list,
        #     episode_descr_config=variable_ways_shots,
        #     split=SPLIT,
        #     image_size=84,
        #     shuffle_buffer_size=300)


        use_bilevel_ontology_list = [False]*len(self.all_datasets)
        use_dag_ontology_list = [False]*len(self.all_datasets)
        # Enable ontology aware sampling for Omniglot and ImageNet. 
        # use_bilevel_ontology_list[5] = True
        # use_dag_ontology_list[4] = True

        # use_bilevel_ontology_list = [True]
        # use_dag_ontology_list = [False]

        variable_ways_shots = config.EpisodeDescriptionConfig(
            num_query=None, num_support=None, num_ways=self.n_ways)

        all_dataset_specs = []
        for dataset_name in self.all_datasets:
          dataset_records_path = os.path.join(self.basepath, dataset_name)
          dataset_spec = dataset_spec_lib.load_dataset_spec(dataset_records_path)
          all_dataset_specs.append(dataset_spec)

        self.episode_generator = pipeline.make_multisource_episode_pipeline(
            dataset_spec_list=all_dataset_specs,
            use_dag_ontology_list=use_dag_ontology_list,
            use_bilevel_ontology_list=use_bilevel_ontology_list,
            episode_descr_config=variable_ways_shots,
            split=self.learning_spec_split,
            image_size=84,
            shuffle_buffer_size=300)


        # self.all_samplers = []
        # for dataset_name in self.all_datasets:
        #     dataset_records_path = os.path.join(self.basepath, dataset_name)
        #     dataset_spec = dataset_spec_lib.load_dataset_spec(dataset_records_path)
        #     self.all_samplers.append(
        #         pipeline.make_one_source_episode_pipeline(
        #             dataset_spec = dataset_spec,
        #             use_bilevel_ontology = self.use_bilevel_ontology,
        #             use_dag_ontology = self.use_dag_ontology,
        #             split = self.learning_spec_split,
        #             image_size = 84, 
        #             episode_descr_config = self.config))

        
        # functions for tf -> torch conversion
        self.to_torch_labels = lambda a: torch.from_numpy(a.numpy()).long()
        self.to_torch_imgs = lambda a: torch.from_numpy(np.transpose(a.numpy(), (0, 3, 1, 2)))



    @ staticmethod
    def iterate_dataset(dataset, n):
        if not tf.executing_eagerly():
            iterator = dataset.make_one_shot_iterator()
            next_element = iterator.get_next()
            with tf.Session() as sess:
                for idx in range(n):
                    yield idx, sess.run(next_element)
        else:
            for idx, episode in enumerate(dataset):
                if idx == n:
                    break
                yield idx, episode


    def sample(self, n_batches, dataset_fixed):
        for i, (e, _) in enumerate(dataset_fixed):
            if i == n_batches:
                break
            return self.to_torch_imgs(e[0]), self.to_torch_labels(e[1]),\
                   self.to_torch_imgs(e[3]), self.to_torch_labels(e[4])


    def __getitem__(self, dataset_id):
        # _, (episode, _) = next(self.iterate_dataset(self.all_samplers[dataset_id], 1))
        # support_images=self.to_torch_imgs(episode[0])
        # support_class_ids=self.to_torch_labels(episode[1])
        # query_images=self.to_torch_imgs(episode[3])
        # query_class_ids=self.to_torch_labels(episode[4])
        # return support_images, support_class_ids, query_images, query_class_ids
        return self.sample(1, self.episode_generator)


    def __len__(self):
        return len(self.all_datasets)




class MultipleMetaDatasets(torch.utils.data.Dataset):

    def __init__(self, support_class_images_set,
                       query_class_images_set,
                       image_size,
                       support_aug, query_aug,
                       fix_support,
                       save_folder,
                       fix_support_path='',
                       verbose=True):

        assert support_class_images_set.keys() == query_class_images_set.keys(),\
            f"""support and query datasets not matching
                support: {support_class_images_set.keys()},
                query: {query_class_images_set.keys()}"""
        
        self.datasets = {}
        for dataset_name in support_class_images_set.keys():
            self.datasets[dataset_name] = MetaDataset(
                       dataset_name,
                       support_class_images_set=support_class_images_set[dataset_name],
                       query_class_images_set=query_class_images_set[dataset_name],
                       image_size=image_size,
                       support_aug=support_aug, query_aug=query_aug,
                       fix_support=fix_support,
                       save_folder=save_folder,
                       fix_support_path=fix_support_path,
                       verbose=verbose
            )

    def __getitem__(self, task_class_info):
        """return a random support, query (input, label) tuple of class cl for a specific dataset

        Args:
            task_class_info (dict): a dictionary containing information of the class requested
                                ['dataset_idx']: index of the dataset
                                ['task_idx']: the index of the task
                                ['cl']: the unique class index
                                ['n_shot']: number of shots requested
                                ['n_query']: number of query requested
                                ['cl_label']: the label to be used for this class

        Returns:
                dict: with keys
                    'task_idx': the index of the task in the batch of tasks for assembling
                    'support_x_cl': tensor of shape (task_class_info['n_shot], c, h, w)
                    'support_y': tensor of shape (task_class_info['n_shot])
                    'query_x_cl': tensor of shape (task_class_info['n_shot], c, h, w)
                    'query_y': tensor of shape (task_class_info['n_shot])
                    'cl': the unique cl identifier (for debugging)
        """

        return self.datasets[task_class_info['dataset_idx']][task_class_info] 

    def __len__(self):
        return sum(len(self.datasets[x].support_class_images_set) for x in self.datasets)



    def save_fixed_support(self, save_folder):
        for dataset_name in self.datasets:
            dataset_save_folder = os.path.join(save_folder, dataset_name)
            if not os.path.exists(dataset_save_folder):
                os.makedir(dataset_save_folder)
            self.datasets[dataset_name].save_fixed_support(dataset_save_folder)
        

    def load_fixed_support(self, fix_support_path):
        for dataset_name in self.datasets:
            dataset_fix_support_path = os.path.join(fix_support_path, dataset_name)
            assert os.path.exists(dataset_fix_support_path)
            self.datasets[dataset_name].load_fixed_support(dataset_fix_support_path)






class MetaDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_name,
                       support_class_images_set,
                       query_class_images_set,
                       image_size,
                       support_aug, query_aug,
                       fix_support,
                       save_folder,
                       fix_support_path='',
                       verbose=True):
        """[summary]

        Args:
            dataset_name (str): name of the dataset (to configure the type of data augmentation)
            support_class_images_set (ClassImagesSet): a data structure that contains the support images of each class
            query_class_images_set (ClassImagesSet): a data structure that contains the query images of each class
            image_size (int): the side length of the square image
            support_aug (bool): whether to use data augmentation for support set
            query_aug (bool): whether to use data augmentation for query set
            save_folder (str): the folder to save the fixed support set information at
            fix_support (int): the fixed number of examples for each class
                               if == 0, then use all of the examples
            randomize_query (bool): whether to use a random number of query examples per class
            fix_support_path (str, optional): the full path location of where the fix support
                                              information is saved at. Load this support set
                                              when this path is not the empty string. Defaults to ''.
            verbose (bool, optional): print the configuration. Defaults to True.
        """
        self.dataset_name = dataset_name
        self.support_class_images_set = support_class_images_set
        self.query_class_images_set = query_class_images_set
        self.image_size = image_size
        # self.n_shot = n_shot
        # self.n_query = n_query
        self.support_aug = support_aug
        self.query_aug = query_aug
        self.fix_support = fix_support
        # self.randomize_query = randomize_query

        # support and query class images set should have the same set of classes
        # although they can have different set of images for each class
        # in most cases support_class_images_set == query_class_images_set
        # except in the case of base_test_acc evaluation using fixed support set

        assert self.support_class_images_set.keys() == self.query_class_images_set.keys(),\
            f"""support and query classes not matching
                support: {self.support_class_images_set.keys()},
                query: {self.query_class_images_set.keys()}"""

        self.classes = list(support_class_images_set.keys())
        # logs
        if verbose:
            print(f"No. of classes in set support {len(self.support_class_images_set)} \
                query {len(self.query_class_images_set)}")
            print("Support set is fixed:", self.fix_support!=0)
            if self.fix_support != 0:
                print("Size of fixed support:", self.fix_support)    
            print("support aug:", support_aug, "query aug:", query_aug)

        # transforms
        self.trans_loader = TransformLoader(image_size)
        support_transform = self.trans_loader.get_composed_transform(dataset_name, aug=support_aug)
        query_transform = self.trans_loader.get_composed_transform(dataset_name, aug=query_aug)
    
        # support
        self.support_sub_dataloader = {} 
        for cl in self.support_class_images_set:
            if verbose:
                print("Setting support loader for class", cl, end =" ")

            sub_dataset = SubMetadataset(
                            class_images=self.support_class_images_set[cl], 
                            n_images=self.fix_support, 
                            cl=cl, 
                            transform=support_transform,
                            target_transform=identity, # likely not needed
                            verbose=verbose)
            self.support_sub_dataloader[cl] = sub_dataset

        # query
        self.query_sub_dataloader = {} 

        for cl in self.query_class_images_set:
            if verbose: 
                print("Setting query loader for class", cl, end=" ")

            sub_dataset = SubMetadataset(
                            class_images=self.query_class_images_set[cl], 
                            n_images=0, 
                            cl=cl,
                            transform=query_transform,
                            target_transform=identity,
                            verbose=verbose)

            self.query_sub_dataloader[cl] = sub_dataset

        # load from fix support path
        if fix_support_path != '':
            self.load_fixed_support(fix_support_path)

        # save fixed support
        if self.fix_support:
            self.save_fixed_support(save_folder)


    def __getitem__(self, task_class_info):
        """return a random support, query (input, label) tuple of class cl

        Args:
            task_class_info (dict): a dictionary containing information of the class requested
                                ['task_idx']: the index of the task
                                ['cl']: the unique class index
                                ['n_shot']: number of shots requested
                                ['n_query']: number of query requested
                                ['cl_label']: the label to be used for this class

        Returns:
                dict: with keys
                    'task_idx': the index of the task in the batch of tasks for assembling
                    'support_x_cl': tensor of shape (task_class_info['n_shot], c, h, w)
                    'support_y': tensor of shape (task_class_info['n_shot])
                    'query_x_cl': tensor of shape (task_class_info['n_shot], c, h, w)
                    'query_y': tensor of shape (task_class_info['n_shot])
                    'cl': the unique cl identifier (for debugging)
        """
        cl = task_class_info['cl']
        result = {'task_idx': task_class_info['task_idx'],
                  'cl': cl}

        if task_class_info['n_shot'] > 0:
            support_x, support_y = self.support_sub_dataloader[cl].get_random_batch(
                                        class_info={
                                            'num': task_class_info['n_shot'],
                                            'cl_label': task_class_info['cl_label'],
                                        })
            result['support_x_cl'] = support_x
            result['support_y_cl'] = support_y
        if task_class_info['n_query'] > 0:
            query_x, query_y = self.query_sub_dataloader[cl].get_random_batch(
                                        class_info={
                                            'num': task_class_info['n_query'],
                                            'cl_label': task_class_info['cl_label'],
                                        })
            result['query_x_cl'] = query_x
            result['query_y_cl'] = query_y

        return result


    def __len__(self):
        return len(self.support_class_images_set)


    def save_fixed_support(self, save_folder):
        """save a dictionary mapping the class_id to a list of paths of the images used in that class's fixed support

        Args:
            save_folder (str): the folder to save this dictionary
        """
        save_path = os.path.join(save_folder, "fixed_support_pool.pkl")
        self.fixed_support_pool  = {}
        for cl in self.support_sub_dataloader:
            cl_dataset = self.support_sub_dataloader[cl]
            fixed_indices = cl_dataset.indices
            self.fixed_support_pool[cl] = [
                self.support_class_images_set[cl].sub_meta[idx] for idx in fixed_indices
            ]

        print("Saving fixed support pool to path", save_path)
        with open(save_path, 'wb') as f:
            torch.save(self.fixed_support_pool, f)


    def load_fixed_support(self, fix_support_path):
        """load a dictionary mapping the class_id to a list of paths of the images used in that class's fixed support
        load the dictionary mapping and set each of ClassImages to be sampling this fixed set (by changing indices)

        Args:
            fix_support_path (str): the exact path to load the dictionary
        """
        print("Loading fixed support pool from path", fix_support_path)
        with open(fix_support_path, 'rb') as f:
            self.fixed_support_pool = torch.load(f)

        for cl in self.fixed_support_pool:
            cl_dataset = self.support_sub_dataloader[cl]
            fixed_images = self.fixed_support_pool[cl]
            fixed_indices = [
                self.support_class_images_set[cl].inv_sub_meta[path] for path in fixed_images
            ]
            cl_dataset.indices = fixed_indices
            print(f"Loading fix support for {cl} with indices {fixed_indices}")


class SubMetadataset(torch.utils.data.Dataset):

    def __init__(self,
                 class_images,
                 cl,
                 n_images=0,
                 transform=transforms.ToTensor(),
                 target_transform=identity,
                 verbose=True):
        """the dataset for a specific class with covariate (input) transformation and variate (label) transformation

        Args:
            class_images (ClassImages): the dataset for this class
            cl (int): the unique index for this class
            n_images (int, optional): the number of images in this Class. if n_images == 0,
                                      then use all the images in the dataset. Defaults to 0.
            transform (torch transform object, optional): the callable torch transformation to be applied to the input. 
                                                          Defaults to transforms.ToTensor().
            target_transform (calleable object, optional): the label transformation. Defaults to identity.
            verbose (bool, optional): Defaults to True.
        """
        self.class_images = class_images
        self.cl = cl 
        """
        Use all indices if n_images=0
        else use a fixed set of indices => fixed support
        """
        if n_images == 0:
            self.indices = np.arange(len(class_images))
        else:
            # only use n_images (indices fixed afterwards)
            self.indices = np.random.choice(len(class_images), size=n_images, replace=False)

        if verbose:
            print(f"using {len(self.indices)} images from class"),

        self.transform = transform
        self.target_transform = target_transform


    def __getitem__(self, i):
        """get ith item of this class's transformed input and transformed label

        Args:
            i (int): integer between 0 and len(self) - 1 that specifies a unique image

        Returns:
            (tuple): transformed img, transformed target
        """        
        # fetch img
        img = self.class_images[self.indices[i]]
        # transforms
        img = self.transform(img)
        target = self.target_transform(self.cl)
        return img, target


    def get_random_batch(self, class_info):
        """get a random batch of data from this submetadataset

        Args:
            class_info (dict): a dictionary containing information of the class requested
                                ['num']: number of examples requested
                                ['cl_label']: the label to be used for this class

        Returns:
            2-element tuple: inputs, labels
                             inputs of shape (class_info['num'], c, h, w)
                             labels of shape (class_info['num']) of integer labels specific by
                                        class_info['cl_label']
        """        

        if class_info['num'] == 0:
            # return None if not requesting
            return None, None

        inputs = \
            [self.transform(self.class_images[idx])
                for idx in np.random.choice(
                                a=self.indices,
                                size=class_info['num'],
                                replace=True)] # Since for some meta-dataset classes \
                                               # class_info['num'] > len(self.indices)

        labels = [self.target_transform(class_info['cl_label'])] * class_info['num']

        return torch.stack(tensors=inputs, dim=0), torch.tensor(labels)


    def __len__(self):
        return len(self.indices)


class ClassImagesSet:

    def __init__(self, *data_files, preload=False):
        """data structure that holds each class's image dataset in the dictionary support_class_images_set/query_class_images_set

        Args:
            data_files (str): variable len argument to paths of the json files.
            preload (bool, optional): whether preload the images into memory. Defaults to False.
        """

        # read json file
        self.meta = {}
        # merge multiple json files (this requires that 'image_labels' to be distinct for different classes)
        for data_file in data_files:
            print("loading image paths, labels from json ", data_file)
            with open(data_file, 'r') as f:
                self.update_meta(json.load(f))

        # map class labels to unique integers in 0, ..., num_unique_classes - 1
        self.label2target = {v:k for k,v in enumerate(np.unique(self.meta['image_labels']))}
        self.target2label = {v:k for k,v in self.label2target.items()}
        # list of unique class labels in dataset
        self.classes = np.unique(self.meta['image_labels']).tolist()

        # fetch all image paths for each class
        self.per_class_image_paths = defaultdict(list)
        for path, cl in zip(self.meta['image_names'], self.meta['image_labels']):
            self.per_class_image_paths[cl].append(path)
        
        # create class images set
        self.class_images_set = {}
        for cl in self.classes:
            self.class_images_set[cl] = ClassImages(self.per_class_image_paths[cl], cl, preload)


    def update_meta(self, json_obj):
        """
        updates self.meta using new keys and values found in
        given json_obj
        """
        
        for key, value in json_obj.items():
            assert type(value) == list, "json file has a value that is not a list"
            """
            the above assertion is necessary for the logic below to 
            to update self.meta. The expected structure of json_obj is
            {k1:v1, k2:v2 ... } where v1, v2 ... are lists
            """
            if key not in self.meta:
                self.meta[key] = value
            else:
                self.meta[key].extend(value)


    def __len__(self):
        """return the number of classes

        Returns:
            int : number of classes in this set
        """
        return len(self.class_images_set)


    def __iter__(self):
        # return iterator of the class indices
        return iter(self.class_images_set)


    def __getitem__(self, cl):
        # return ClassImages object for class cl
        return self.class_images_set[cl]


    def items(self):
        # return (cl, ClassImages) tuple for every class cl
        for cl in self.class_images_set:
            yield cl, self.class_images_set[cl]

    def keys(self):
        # return classes, mainly to interface this class as a dict
        return self.classes


class ClassImages:

    def __init__(self, sub_meta, cl, preload=False):
        """the dataset containing all the images of a specific class cl, examples obtainable by __getitem__(i):

        Args:
            sub_meta (list of str): a list of image paths of class name cl
            cl (int): a unique integer identifying the class
            preload (bool, optional): whether to load all the images into memory. Defaults to False.
        """
        self.sub_meta = sub_meta
        self.inv_sub_meta = {v:k for k,v in enumerate(self.sub_meta)} # maps the unique file path to an index
        self.images = []
        self.cl = cl 
        self.preload = preload
        
        if preload:
            print(f"Attempt loading class {cl} into memory")
            # with tqdm.tqdm(total=len(self.sub_meta)) as pbar_memory_load:
            with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
                # Process the list of files, but split the work across the process pool to use all CPUs!
                for image in executor.map(load_image, self.sub_meta):
                    self.images.append(image)
            print(f"Done loading class {cl} into memory -- found {len(self.images)} images")


        self.original_sub_meta = self.sub_meta
        self.original_images = self.images


    def __getitem__(self, i):
        # load the i-th image of this class
        if self.preload:
            img = self.images[i]
        else:
            img = load_image(self.sub_meta[i])
        return img


    def __len__(self):
        return len(self.sub_meta)


    def resample_images(self, n_chosen):
        """
        Randomly choose n_chosen objects out of original sub_meta to
        create new sub_meta. Basically, it reduces the images in a class.  
        """
        assert n_chosen > 0, "Must select non zero examples for each class"
        selected_indices = np.random.choice(len(self.original_sub_meta), n_chosen, replace=False)
        self.sub_meta = [self.original_sub_meta[x] for x in selected_indices]
        if self.preload:
            self.images = [self.original_images[x] for x in selected_indices]
        print(f"No. of samples in class {self.cl}: {len(self.sub_meta)}")


class SimpleDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_name,
                       class_images_set,
                       image_size,
                       aug,
                       verbose=True,
                       sample=0):

        """[summary]

        Args:
            dataset_name (str): name of the dataset (to configure the type of data augmentation)
            class_images_set (ClassImagesSet): a data structure that contains the images of each class
            image_size (int): the side length of the square image
            aug (bool): whether to use data augmentation for support set
            verbose (bool, optional): print the configuration. Defaults to True.
            sample (int, optional): if sample > 0, each class samples "sample" number of examples from the given ClassImages
        """
        self.dataset_name = dataset_name
        self.class_images_set = class_images_set
        self.image_size = image_size
        self.aug = aug
        self.sample = sample
        
        # list of classes
        self.classes = list(class_images_set.keys())
        
        # logs
        if verbose:
            print(f"No. of classes in dataset {len(self.class_images_set)}")
            print("Augmentation:", aug)

        # transforms
        self.trans_loader = TransformLoader(image_size)
        self.transform = self.trans_loader.get_composed_transform(dataset_name, aug=aug)
        
        # create a single list of all images and their labels
        # this is a concatenated list of indices within each class in class_images_set
        self.indices_within_each_class_images_set = []
        self.class_labels = []
        for i, cl in enumerate(self.classes):
            if self.sample > 0:
                sampled_indices = np.random.choice(
                    np.arange(len(self.class_images_set[cl])), self.sample, replace=False)
            else:
                sampled_indices = np.arange(len(self.class_images_set[cl]))
            self.indices_within_each_class_images_set.extend(sampled_indices)
            self.class_labels.extend([i] * len(sampled_indices))

        assert len(self.class_labels) == len(self.indices_within_each_class_images_set)

        # n_images in total
        self.total_images = len(self.indices_within_each_class_images_set)
        print("Total images", self.total_images)
        

    def __getitem__(self, i):
        class_label = self.class_labels[i]
        class_images = self.class_images_set[self.classes[class_label]]
        img = class_images[self.indices_within_each_class_images_set[i]]
        transformed_img = self.transform(img)
        return transformed_img, class_label


    def __len__(self):
        # sum all images for each ClassImages object in ClassImagesSet
        return self.total_images


if __name__ == '__main__':

    cis = \
    ClassImagesSet(data_file='/home/anonymous/projects/meta-analysis-classification/data/new_miniimagenet/val.json',
                   preload=False)
    print(cis.class_images_set)
