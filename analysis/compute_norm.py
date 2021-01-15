import os
from tqdm import tqdm
import json
import argparse
import pickle
import torch
import torch.nn as nn
import numpy as np
import pprint
from tensorboardX import SummaryWriter
import re
import shutil
from datetime import datetime


from src.models import shallow_conv, resnet_12, wide_resnet, dense_net
from src.algorithm_trainer.algorithm_trainer import Meta_algorithm_trainer, Init_algorithm_trainer
from src.algorithms.algorithm import SVM, ProtoNet, Ridge, InitBasedAlgorithm
from src.optimizers import modified_sgd
from src.data.dataset_managers import MetaDataLoader
from src.data.datasets import MetaDataset, ClassImagesSet




def str2bool(arg):
    return arg.lower() == 'true'


def l2_norm(model):

    val = 0.
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, torch.norm(param, p=2))
            val += torch.sum(param * param)
    return torch.sqrt(val)


def compute_norm(args, p=2):


    ####################################################
    #             MODEL/BACKBONE CREATION              #
    ####################################################
    
    print("\n", "--"*20, "MODEL", "--"*20)
    dataset_name = args.dataset_name
    if args.model_type == 'resnet_12':
        if 'miniImagenet' in dataset_name or 'CUB' in dataset_name:
            model = resnet_12.resnet12(avg_pool=str2bool(args.avg_pool), drop_rate=0.1, dropblock_size=5,
                num_classes=args.num_classes_train, classifier_type=args.classifier_type,
                projection=str2bool(args.projection))
        else:
            model = resnet_12.resnet12(avg_pool=str2bool(args.avg_pool), drop_rate=0.1, dropblock_size=2,
                num_classes=args.num_classes_train, classifier_type=args.classifier_type,
                projection=str2bool(args.projection))
    elif args.model_type in ['conv64', 'conv48', 'conv32']:
        dim = int(args.model_type[-2:])
        model = shallow_conv.ShallowConv(z_dim=dim, h_dim=dim, num_classes=args.num_classes_train, 
            classifier_type=args.classifier_type, projection=str2bool(args.projection))
    elif args.model_type == 'wide_resnet28_10':
        model = wide_resnet.wrn28_10(
            projection=str2bool(args.projection), classifier_type=args.classifier_type)
    elif args.model_type == 'wide_resnet16_10':
        model = wide_resnet.wrn16_10(
            projection=str2bool(args.projection), classifier_type=args.classifier_type)
    else:
        raise ValueError(
            'Unrecognized model type {}'.format(args.model_type))
    print("Model\n" + "=="*27)    
    print(model)   



    ####################################################
    #                LOAD FROM CHECKPOINT              #
    ####################################################

    if args.checkpoint != '':
        print(f"loading model from {args.checkpoint}")
        model_dict = model.state_dict()
        chkpt = torch.load(args.checkpoint, map_location=torch.device('cpu'))
        try:
            print(f"loading optimizer from {args.checkpoint}")
            optimizer.state = chkpt['optimizer'].state
            print("Successfully loaded optimizer")
        except:
            print("Failed to load optimizer")
        chkpt_state_dict = chkpt['model']
        chkpt_state_dict_cpy = chkpt_state_dict.copy()
        # remove "module." from key, possibly present as it was dumped by data-parallel
        for key in chkpt_state_dict_cpy.keys():
            if 'module.' in key:
                new_key = re.sub('module\.', '',  key)
                chkpt_state_dict[new_key] = chkpt_state_dict.pop(key)
        chkpt_state_dict = {k: v for k, v in chkpt_state_dict.items() if k in model_dict}
        model_dict.update(chkpt_state_dict)
        updated_keys = set(model_dict).intersection(set(chkpt_state_dict))
        print(f"Updated {len(updated_keys)} keys using chkpt")
        print("Following keys updated :", "\n".join(sorted(updated_keys)))
        missed_keys = set(model_dict).difference(set(chkpt_state_dict))
        print(f"Missed {len(missed_keys)} keys")
        print("Following keys missed :", "\n".join(sorted(missed_keys)))
        model.load_state_dict(model_dict)
                    

    ####################################################
    #                COMPUTE NORM HERE                 #
    ####################################################
     

    if p==2:
        return l2_norm(model)
    else:
        raise ValueError('Unspecified norm type') 
    
        


if __name__ == '__main__':

    
    parser = argparse.ArgumentParser(
        description='Training the feature backbone on all classes from all tasks.')    
    parser.add_argument('--model-type', type=str, default='gatedconv',
        help='type of the model')
    parser.add_argument('--dataset-name', type=str, default='miniImagenet',
        help='dataset name')
    parser.add_argument('--classifier-type', type=str, default='no-classifier',
        help='classifier type [distance based, linear, GDA]')
    parser.add_argument('--checkpoint', type=str, default='',
        help='path to saved parameters.')
    parser.add_argument('--classifier-metric', type=str, default='euclidean',
        help='')
    parser.add_argument('--projection', type=str, default='False',
        help='')
    parser.add_argument('--avg-pool', type=str, default='True',
        help='')
    parser.add_argument('--num-classes-train', type=int, default=0,
        help='no of train classes')
    
    args = parser.parse_args()
    l2_norm = compute_norm(args, p=2)

    print("l2_norm", l2_norm)
