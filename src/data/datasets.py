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


class MetaDataset:

    def __init__(self, dataset_name, class_images, image_size, n_shot, n_query, 
        support_aug, query_aug, save_folder, fix_support, randomize_query, fix_support_path='', verbose=True):
        
        self.class_images = class_images
        self.fix_support = fix_support
        self.n_shot = n_shot
        self.n_query = n_query
        self.class_images = class_images
        self.support_aug = support_aug
        self.query_aug = query_aug
        self.randomize_query = randomize_query
        
        # logs
        if verbose:
            print("No. of classes in set", len(self.class_images))
            print("Support set is fixed:", self.fix_support!=0)
            if self.fix_support!=0:
                print("Size of fixed support:", self.fix_support)    
            print("Size of Support:", self.n_shot)
            print("Size of Query:", self.n_query, "randomize query", self.randomize_query)
            print("support aug:", support_aug, "query aug:", query_aug)

        # transforms
        self.trans_loader = TransformLoader(image_size)
        support_transform = self.trans_loader.get_composed_transform(dataset_name, support_aug)
        query_transform = self.trans_loader.get_composed_transform(dataset_name, query_aug)
    
        # support
        if n_shot > 0:
            self.support_sub_dataloader = {} 
            support_sub_data_loader_params = dict(batch_size = n_shot,
                shuffle = True,
                num_workers = 0, #use main thread only or may receive multiple batches
                pin_memory = False)        
            for cl in self.class_images.class_images_set:
                if verbose:
                    print("Setting support loader for class", self.class_images.target2label[cl], end =" ")
                sub_dataset = SubMetadataset(
                    self.class_images.class_images_set[cl], 
                    n_images=self.fix_support, 
                    cl=cl, 
                    transform=support_transform,
                    verbose=verbose)
                self.support_sub_dataloader[cl]=torch.utils.data.DataLoader(
                    sub_dataset, **support_sub_data_loader_params)

        # query
        if n_query > 0:
            self.query_sub_dataloader = {} 
            
            # double query points if randomize_query is True. Sample more, permute and select top K 
            # This is a pseudo method for inducing some variance. To increase this variance
            # we can triple the query points instead of doubling them. also make corresponding change in trainer.

            query_sub_data_loader_params = dict(batch_size = 2*n_query if self.randomize_query else n_query,
                shuffle = True,
                num_workers = 0, #use main thread only or may receive multiple batches
                pin_memory = False)        
            for cl in self.class_images.class_images_set:
                if verbose: 
                    print("Setting query loader for class", self.class_images.target2label[cl], end =" ")
                sub_dataset = SubMetadataset(
                    self.class_images.class_images_set[cl], 
                    n_images=0, 
                    cl=cl, 
                    transform=query_transform,
                    verbose=verbose)
                self.query_sub_dataloader[cl]=torch.utils.data.DataLoader(
                    sub_dataset, **query_sub_data_loader_params)

        # load from fix support path    
        if fix_support_path != '':
            self.load_fixed_support(fix_support_path)

        # save fixed support
        if self.fix_support:
            self.save_fixed_support(save_folder)

        

    def __getitem__(self,i):
        if self.n_shot > 0 and self.n_query == 0:
            return next(iter(self.support_sub_dataloader[i]))
        elif self.n_query > 0 and self.n_shot == 0:
            return next(iter(self.query_sub_dataloader[i]))
        elif self.n_shot > 0 and self.n_query > 0:
            return torch.cat([next(iter(self.support_sub_dataloader[i]))[0], next(iter(self.query_sub_dataloader[i]))[0]], dim=0), \
                 torch.cat([next(iter(self.support_sub_dataloader[i]))[1], next(iter(self.query_sub_dataloader[i]))[1]], dim=0)
        else:
            return None


    def save_fixed_support(self, save_folder):

        save_path = os.path.join(save_folder, "fixed_support_pool.pkl")
        self.fixed_support_pool  = {}
        for cl in self.support_sub_dataloader:
            class_id = self.class_images.target2label[cl] 
            cl_dataset = self.support_sub_dataloader[cl].dataset
            fixed_indices = cl_dataset.indices
            self.fixed_support_pool[cl] = [
                self.class_images.class_images_set[cl].sub_meta[idx] for idx in fixed_indices]

        print("Saving fixed support pool to path", save_path)
        with open(save_path, 'wb') as f:
            torch.save(self.fixed_support_pool, f)


    def load_fixed_support(self, fix_support_path):

        print("Loading fixed support pool from path", fix_support_path)
        with open(fix_support_path, 'rb') as f:
            self.fixed_support_pool = torch.load(f)

        for class_id in self.fixed_support_pool:
            cl = self.class_images.label2target[class_id] 
            cl_dataset = self.support_sub_dataloader[cl].dataset
            fixed_images = self.fixed_support_pool[class_id]
            fixed_indices = [
                self.class_images.class_images_set[cl].inv_sub_meta[path] for path in fixed_images]
            cl_dataset.indices = fixed_indices
            print(f"Loading fix support for {class_id} with indices {fixed_indices}")



    def __len__(self):
        return len(self.class_images)




class SubMetadataset:

    def __init__(self, class_images, cl, n_images=0,
        transform=transforms.ToTensor(), target_transform=identity, verbose=True):

        self.class_images = class_images
        """
        Use all indices if n_images=0 => fixed support
        else use a fixed set of indices
        """
        self.indices = np.random.choice(len(class_images), n_images, replace=False)\
             if n_images else np.arange(len(class_images))
        if verbose:
            print(f"using {len(self.indices)} images from class"),
        self.cl = cl 
        self.transform = transform
        self.target_transform = target_transform
        

    def __getitem__(self,i):
        # fetch img
        img = self.class_images[self.indices[i]]
        # transforms
        img = self.transform(img)
        target = self.target_transform(self.cl)
        return img, target

    def __len__(self):
        return len(self.indices)




class ClassImagesSet:

    def __init__(self, data_file, preload=False):
        
        # read json file
        with open(data_file, 'r') as f:
            self.meta = json.load(f)

        # map class labels to unique integers
        self.label2target = {v:k for k,v in enumerate(np.unique(self.meta['image_labels']))}
        self.target2label = {v:k for k,v in self.label2target.items()}
        self.meta['image_labels'] = list(map(
            lambda x: self.label2target[x], self.meta['image_labels']))
 
        # list of class labels
        self.classes = np.unique(self.meta['image_labels']).tolist()
        
        # fetch all image paths for each class
        self.per_class_image_paths = defaultdict(list)
        for x,y in zip(self.meta['image_names'],self.meta['image_labels']):
            self.per_class_image_paths[y].append(x)
        
        # create class images set
        self.class_images_set = {}
        for cl in self.classes:
            self.class_images_set[cl] = ClassImages(self.per_class_image_paths[cl], cl, preload)
    

    def __len__(self):
        return len(self.class_images_set)



class ClassImages:

    def __init__(self, sub_meta, cl, preload=False):

        self.sub_meta = sub_meta
        self.inv_sub_meta = {v:k for k,v in enumerate(self.sub_meta)}
        self.images = []
        self.cl = cl 
        self.preload = preload
        
        if preload:
            print("Attempt loading class {} into memory".format(cl))
            # with tqdm.tqdm(total=len(self.sub_meta)) as pbar_memory_load:
            with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
                # Process the list of files, but split the work across the process pool to use all CPUs!
                for image in executor.map(load_image, self.sub_meta):
                    self.images.append(image)
            print("Done loading class {} into memory -- found {} images".format(cl, len(self.images)))
                        

    def __getitem__(self,i):
        if self.preload:
            img = self.images[i]
        else:
            img = Image.open(self.sub_meta[i]).convert('RGB')
        return img

    def __len__(self):
        return len(self.sub_meta)



