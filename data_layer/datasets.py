import torch
from PIL import Image
import json
import numpy as np
import torchvision.transforms as transforms
import concurrent.futures
import tqdm
import os
identity = lambda x:x




class ClassicalDataset:

    def __init__(self, data_file, transform, target_transform=identity):
        with open(data_file, 'r') as f:
            self.data_file = json.load(f)
        self.transform = transform
        self.target_transform = target_transform
        self.label2target = {v:k for k,v in enumerate(np.unique(self.data_file['image_labels']))}
        self.data_file['image_labels'] = list(map(
            lambda x: self.label2target[x], self.data_file['image_labels']))

    def __getitem__(self,i):
        image_path = os.path.join(self.data_file['image_names'][i])
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        target = self.target_transform(self.data_file['image_labels'][i])
        return img, target

    def __len__(self):
        return len(self.data_file['image_names'])




class MetaDataset:

    def __init__(self, data_file, n_shot, n_query, support_transform, query_transform, fix_support=0):
        
        self.fix_support = fix_support
        print("Support set is fixed:", self.fix_support!=0)
        self.n_shot = n_shot
        self.n_query = n_query

        with open(data_file, 'r') as f:
            self.meta = json.load(f)

        self.label2target = {v:k for k,v in enumerate(np.unique(self.meta['image_labels']))}
        self.meta['image_labels'] = list(map(
            lambda x: self.label2target[x], self.meta['image_labels']))
 
        self.cl_list = np.unique(self.meta['image_labels']).tolist()

        self.sub_meta = {}
        self.class_images = {}
        for cl in self.cl_list:
            self.sub_meta[cl] = []

        for x,y in zip(self.meta['image_names'],self.meta['image_labels']):
            self.sub_meta[y].append(x)
            
        for cl in self.cl_list:
            self.class_images[cl] = ClassImages(self.sub_meta[cl], cl)

        if n_shot > 0:
            self.support_sub_dataloader = [] 
            support_sub_data_loader_params = dict(batch_size = n_shot, # (n_support) * n_tasks
                                    shuffle = True,
                                    num_workers = 0, #use main thread only or may receive multiple batches
                                    pin_memory = False)        
            for cl in self.cl_list:
                print("Setting support loader for class", cl)
                sub_dataset = SupportOrQueryForSubMetadataset(self.class_images[cl], 
                    n_images=self.fix_support, cl=cl, transform=support_transform)
                self.support_sub_dataloader.append(
                    torch.utils.data.DataLoader(sub_dataset, **support_sub_data_loader_params))

        if n_query > 0:
            self.query_sub_dataloader = [] 
            query_sub_data_loader_params = dict(batch_size = n_query, # (n_query) * n_tasks
                                    shuffle = True,
                                    num_workers = 0, #use main thread only or may receive multiple batches
                                    pin_memory = False)        
            for cl in self.cl_list:
                print("Setting query loader for class", cl)
                sub_dataset = SupportOrQueryForSubMetadataset(self.class_images[cl], cl=cl, transform = query_transform)
                self.query_sub_dataloader.append(
                    torch.utils.data.DataLoader(sub_dataset, **query_sub_data_loader_params))
                

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


    def __len__(self):
        return len(self.cl_list)




def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    return img



class SupportOrQueryForSubMetadataset:

    def __init__(self, class_images, cl, n_images=0,
        transform=transforms.ToTensor(), target_transform=identity):

        self.class_images = class_images
        self.indices = np.random.choice(len(class_images), n_images, replace=False)\
             if n_images else np.arange(len(class_images))
        print(f"using {len(self.indices)} images from class")
        self.cl = cl 
        self.transform = transform
        self.target_transform = target_transform
        

    def __getitem__(self,i):
        # image_path = os.path.join(self.sub_meta[i])
        # img = Image.open(image_path).convert('RGB')
        img = self.class_images[self.indices[i]]
        img = self.transform(img)
        target = self.target_transform(self.cl)
        return img, target

    def __len__(self):
        return len(self.indices)




class ClassImages:

    def __init__(self, sub_meta, cl, printt=False):

        self.sub_meta = sub_meta
        self.images = []
        self.cl = cl 
        self.print=printt


        print("Attempt loading class {} into memory".format(cl))
        # with tqdm.tqdm(total=len(self.sub_meta)) as pbar_memory_load:
        with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
            # Process the list of files, but split the work across the process pool to use all CPUs!
            for image in executor.map(load_image, self.sub_meta):
                self.images.append(image)
                    # pbar_memory_load.update(1)
        print("Done loading class {} into memory -- found {} images".format(cl, len(self.images)))
                        
    

    def __getitem__(self,i):
        # if self.print:
        #     print( '%d -%d' %(self.cl,i), os.path.join(self.sub_meta[i]))
        # image_path = os.path.join(self.sub_meta[i])
        # img = Image.open(image_path).convert('RGB')
        img = self.images[i]
        # img = self.transform(img)
        # target = self.target_transform(self.cl)
        return img

    def __len__(self):
        return len(self.sub_meta)



