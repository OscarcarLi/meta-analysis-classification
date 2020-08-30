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

    def __init__(self, data_file, n_shot, n_query, transform, fix_support=False):
        
        self.fix_support = fix_support
        self.n_shot = n_shot
        self.n_query = n_query

        with open(data_file, 'r') as f:
            self.meta = json.load(f)

        self.label2target = {v:k for k,v in enumerate(np.unique(self.meta['image_labels']))}
        self.meta['image_labels'] = list(map(
            lambda x: self.label2target[x], self.meta['image_labels']))
 
        self.cl_list = np.unique(self.meta['image_labels']).tolist()

        self.sub_meta = {}
        for cl in self.cl_list:
            self.sub_meta[cl] = []

        for x,y in zip(self.meta['image_names'],self.meta['image_labels']):
            self.sub_meta[y].append(x)

        sub_data_loader_params_batch_size = n_query
        if not fix_support:
            sub_data_loader_params_batch_size += n_shot
        
        if sub_data_loader_params_batch_size > 0:
            self.sub_dataloader = [] 
            sub_data_loader_params = dict(batch_size = sub_data_loader_params_batch_size, # (n_support + n_query) * n_tasks
                                    shuffle = True,
                                    num_workers = 0, #use main thread only or may receive multiple batches
                                    pin_memory = False)        
            for cl in self.cl_list:
                sub_dataset = SubMetaDataset(self.sub_meta[cl], cl, transform = transform)
                self.sub_dataloader.append(
                    torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params))

        if fix_support:
            
            self.fix_support_sub_dataloader = [] 
            fix_support_sub_data_loader_params = dict(batch_size = n_shot, # (n_support) * n_tasks
                                    shuffle = False,
                                    num_workers = 0, #use main thread only or may receive multiple batches
                                    pin_memory = False)        
            for cl in self.cl_list:
                fix_support_sub_dataset = SubMetaDataset(self.sub_meta[cl][:n_shot], cl, transform = transform)
                self.fix_support_sub_dataloader.append(
                    torch.utils.data.DataLoader(fix_support_sub_dataset, **fix_support_sub_data_loader_params))
    

    def __getitem__(self,i):
        if self.fix_support and self.n_query > 0:
            return torch.cat([next(iter(self.fix_support_sub_dataloader[i]))[0], next(iter(self.sub_dataloader[i]))[0]], dim=0), \
                 torch.cat([next(iter(self.fix_support_sub_dataloader[i]))[1], next(iter(self.sub_dataloader[i]))[1]], dim=0)
        elif self.fix_support:
            return next(iter(self.fix_support_sub_dataloader[i]))
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



