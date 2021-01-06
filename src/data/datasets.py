import torch
from PIL import Image
import json
import numpy as np
import torchvision.transforms as transforms
import concurrent.futures
from collections import defaultdict
import tqdm
import os

from src.data.transforms import TransformLoader


# for transform
identity = lambda x:x

def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    return img


class MetaDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_name,
                       support_class_images_set,
                       query_class_images_set,
                       image_size,
                       n_shot, n_query, 
                       support_aug, query_aug,
                       save_folder,
                       fix_support,
                       randomize_query,
                       fix_support_path='',
                       verbose=True):
        """[summary]

        Args:
            dataset_name (str): name of the dataset (to configure the type of data augmentation)
            support_class_images_set (ClassImagesSet): a data structure that contains the support images of each class
            query_class_images_set (ClassImagesSet): a data structure that contains the query images of each class
            image_size (int): the side length of the square image
            n_shot (int): number of support examples per class
            n_query (int): number of query examples per class
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
        self.n_shot = n_shot
        self.n_query = n_query
        self.support_aug = support_aug
        self.query_aug = query_aug
        self.fix_support = fix_support
        self.randomize_query = randomize_query

        # support and query class images set should have the same set of classes
        # although they can have different set of images for each class
        # in most cases support_class_images_set == query_class_images_set
        # except in the case of base_test_acc evaluation using fixed support set

        assert self.support_class_images_set.keys() == self.query_class_images_set.keys()
        self.classes = support_class_images_set.keys()

        # logs
        if verbose:
            print(f"No. of classes in set support {len(self.support_class_images_set)} \
                query {len(self.query_class_images_set)}")
            print("Support set is fixed:", self.fix_support!=0)
            if self.fix_support!=0:
                print("Size of fixed support:", self.fix_support)    
            print("Size of Support:", self.n_shot)
            print("Size of Query:", self.n_query, "randomize query", self.randomize_query)
            print("support aug:", support_aug, "query aug:", query_aug)

        # transforms
        self.trans_loader = TransformLoader(image_size)
        support_transform = self.trans_loader.get_composed_transform(dataset_name, aug=support_aug)
        query_transform = self.trans_loader.get_composed_transform(dataset_name, aug=query_aug)
    
        # support
        if n_shot > 0:
            self.support_sub_dataloader = {} 
            support_sub_data_loader_params = dict(batch_size=n_shot,
                                                  shuffle=True,
                                                  num_workers=0, #use main thread only or may receive multiple batches
                                                  pin_memory=False)
            for cl in self.support_class_images_set:
                if verbose:
                    print("Setting support loader for class", cl, end =" ")

                sub_dataset = SubMetadataset(
                                class_images=self.support_class_images_set[cl], 
                                n_images=self.fix_support, 
                                cl=cl, 
                                transform=support_transform,
                                verbose=verbose,
                                target_transform=(lambda x: self.support_class_images_set.label2target[x]))
                self.support_sub_dataloader[cl] = torch.utils.data.DataLoader(
                                                            dataset=sub_dataset, 
                                                            **support_sub_data_loader_params)

        # query
        if n_query > 0:
            self.query_sub_dataloader = {} 
            
            # double query points if randomize_query is True. Sample more, permute and select top K 
            # This is a pseudo method for inducing some variance. To increase this variance
            # we can triple the query points instead of doubling them. also make corresponding change in trainer.

            query_sub_data_loader_params = dict(batch_size=2*n_query if self.randomize_query else n_query,
                                                shuffle=True,
                                                num_workers=0, #use main thread only or may receive multiple batches
                                                pin_memory=False)        

            for cl in self.query_class_images_set:
                if verbose: 
                    print("Setting query loader for class", cl, end=" ")

                sub_dataset = SubMetadataset(
                    class_images=self.query_class_images_set[cl], 
                    n_images=0, 
                    cl=cl, 
                    transform=query_transform,
                    verbose=verbose,
                    target_transform=(lambda x: self.query_class_images_set.label2target[x]))
                self.query_sub_dataloader[cl]=torch.utils.data.DataLoader(
                    sub_dataset, **query_sub_data_loader_params)

        # load from fix support path
        if fix_support_path != '':
            self.load_fixed_support(fix_support_path)

        # save fixed support
        if self.fix_support:
            self.save_fixed_support(save_folder)


    def __getitem__(self, cl):
        """return a random support, query (input, label) tuple of class cl

        Args:
            cl (int): the index of the class

        Returns:
            (input, label): a tuple where input, label are tensors with the first part
                            of the first dimension of the support set, the second
                            part of the first dimension of the query set.
                            for example, input of shape [20, 3, 84, 84]
                                         label of shape [20]
                                         when n_shot == 5 and n_query == 15.
                                         input[:5,:], label[:5,:] support set of this class
                                         input[5:,:], label[5:,:] query set of this class
        """
        if self.n_shot > 0 and self.n_query == 0:
            return next(iter(self.support_sub_dataloader[cl]))
        elif self.n_query > 0 and self.n_shot == 0:
            return next(iter(self.query_sub_dataloader[cl]))
        elif self.n_shot > 0 and self.n_query > 0:
            return torch.cat([next(iter(self.support_sub_dataloader[cl]))[0], next(iter(self.query_sub_dataloader[cl]))[0]], dim=0), \
                   torch.cat([next(iter(self.support_sub_dataloader[cl]))[1], next(iter(self.query_sub_dataloader[cl]))[1]], dim=0)
        else:
            return None


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
            cl_dataset = self.support_sub_dataloader[cl].dataset
            fixed_indices = cl_dataset.indices
            self.fixed_support_pool[cl] = [
                self.support_class_images_set[cl].sub_meta[idx] for idx in fixed_indices]

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
            cl_dataset = self.support_sub_dataloader[cl].dataset
            fixed_images = self.fixed_support_pool[cl]
            fixed_indices = [
                self.support_class_images_set[cl].inv_sub_meta[path] for path in fixed_images]
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
            class_images (ClassImages): the set of images in class cl 
            cl : unique index for the class in its parent class_images_set
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
        # fetch img
        img = self.class_images[self.indices[i]]
        # transforms
        img = self.transform(img)
        target = self.target_transform(self.cl)
        return img, target


    # def get_list_of_items(self, ls):
    #     return \
    #     (torch.stack(tensors=[self.transform(self.class_images[self.indices[i]]) for i in ls],
    #                 dim=0),
    #     torch.stack(tensors=[self.target_transform(self.cl) for _ in ls],
    #                 dim=0))


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
        for x,y in zip(self.meta['image_names'], self.meta['image_labels']):
            self.per_class_image_paths[y].append(x)
        
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
                        

    def __getitem__(self, i):
        # load the i-th image of this class
        if self.preload:
            img = self.images[i]
        else:
            img = load_image(self.sub_meta[i])
        return img

    def __len__(self):
        return len(self.sub_meta)


if __name__ == '__main__':

    cis = ClassImagesSet(
        'datasets/filelists/FC100/base.json', 'datasets/filelists/FC100/novel.json', preload=False)
    print(cis.class_images_set)
