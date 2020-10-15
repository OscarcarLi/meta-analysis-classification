import os
import cifar10.model_loader


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

import sys
sys.path.append('..')

from algorithm_trainer.models import resnet_12

# def load(dataset, model_name, model_file, data_parallel=False):
#     if dataset == 'cifar10':
#         net = cifar10.model_loader.load(model_name, model_file, data_parallel)
#     return net




def load(checkpoint):
    
    model = resnet_12.resnet12(
        avg_pool=True, drop_rate=0.1, dropblock_size=2, no_fc_layer=True, projection=False)
    print(f"loading from {checkpoint}")
    model_dict = model.state_dict()
    chkpt_state_dict = torch.load(checkpoint)
    if 'model' in chkpt_state_dict:
        chkpt_state_dict = chkpt_state_dict['model']
    chkpt_state_dict_cpy = chkpt_state_dict.copy()
    # remove "module." from key, possibly present as it was dumped by data-parallel
    for key in chkpt_state_dict_cpy.keys():
        if 'module.' in key:
            new_key = re.sub('module\.', '',  key)
            chkpt_state_dict[new_key] = chkpt_state_dict.pop(key)
    chkpt_state_dict = {k: v for k, v in chkpt_state_dict.items() if k in model_dict}
    model_dict.update(chkpt_state_dict)
    updated_keys = set(model_dict).intersection(set(chkpt_state_dict))
    missed_keys = set(model_dict).difference(set(chkpt_state_dict))
    print(f"Missed {len(missed_keys)} keys")
    model.load_state_dict(model_dict)
    model.eval()
    return model
