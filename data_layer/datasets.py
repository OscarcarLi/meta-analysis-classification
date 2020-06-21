import torch
from PIL import Image
import json
import numpy as np
import torchvision.transforms as transforms
import os
identity = lambda x:x


class ClassicalDataset:

    def __init__(self, data_file, transform, target_transform=identity):
        with open(data_file, 'r') as f:
            self.data_file = json.load(f)
        self.transform = transform
        self.target_transform = target_transform


    def __getitem__(self,i):
        image_path = os.path.join(self.data_file['image_names'][i])
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        target = self.target_transform(self.data_file['image_labels'][i])
        return img, target

    def __len__(self):
        return len(self.data_file['image_names'])




class MetaDataset:

    def __init__(self, data_file, per_task_batch_size, transform):
        
        with open(data_file, 'r') as f:
            self.meta = json.load(f)
 
        self.cl_list = np.unique(self.meta['image_labels']).tolist()

        self.sub_meta = {}
        for cl in self.cl_list:
            self.sub_meta[cl] = []

        for x,y in zip(self.meta['image_names'],self.meta['image_labels']):
            self.sub_meta[y].append(x)

        self.sub_dataloader = [] 
        sub_data_loader_params = dict(batch_size = per_task_batch_size, # (n_support + n_query) * n_tasks
                                  shuffle = True,
                                  num_workers = 0, #use main thread only or may receive multiple batches
                                  pin_memory = False)        
        for cl in self.cl_list:
            sub_dataset = SubMetaDataset(self.sub_meta[cl], cl, transform = transform)
            self.sub_dataloader.append(
                torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params))

    def __getitem__(self,i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.cl_list)




class SubMetaDataset:

    def __init__(self, sub_meta, cl, 
        transform=transforms.ToTensor(), target_transform=identity):

        self.sub_meta = sub_meta
        self.cl = cl 
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self,i):
        #print( '%d -%d' %(self.cl,i))
        image_path = os.path.join(self.sub_meta[i])
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        target = self.target_transform(self.cl)
        return img, target

    def __len__(self):
        return len(self.sub_meta)



