
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


from algorithm_trainer.models import gated_conv_net_original, resnet
from algorithm_trainer.algorithm_trainer import Generic_adaptation_trainer
from data_layer.dataset_managers import MetaDataManager
from data_layer.datasets import MetaDataset
from analysis.objectives import *


""" Always configure aux func before running analysis.
Find entire list in objectives file. Can be None too
"""

aux_func = var_reduction_disc
    


def main(args):

    is_training = False
    writer = None
    
    # load checkpoint
    if args.model_type == 'resnet':
        model = resnet.ResNet18(num_classes=args.num_classes, 
            distance_classifier=args.distance_classifier, add_bias=args.add_bias)
    elif args.model_type == 'conv64':
        model = ImpRegConvModel(
            input_channels=3, num_channels=64, img_side_len=image_size, num_classes=args.num_classes,
            verbose=True, retain_activation=True, use_group_norm=True, add_bias=args.add_bias)
    else:
        raise ValueError(
            'Unrecognized model type {}'.format(args.model_type))
    print("Model\n" + "=="*27)    
    print(model)
    if args.checkpoint != '':
        print(f"loading from {args.checkpoint}")
        model_dict = model.state_dict()
        chkpt_state_dict = torch.load(args.checkpoint)['model']
        chkpt_state_dict_cpy = chkpt_state_dict.copy()
        if args.no_fc_layer:
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
        model.load_state_dict(model_dict)                
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_number
    print('Using GPUs: ', os.environ["CUDA_VISIBLE_DEVICES"])
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.cuda()
        

    # optimizer
    optimizer = torch.optim.Adam([
        {'params': model.parameters(), 'lr': args.lr},
    ])
    print("Total n_epochs: ", args.n_epochs)

    
    # data loader
    image_size = args.img_side_len
    val_file = os.path.join(args.dataset_path, 'val.json')
    val_datamgr = MetaDataManager(
        image_size, batch_size=args.batch_size_val, n_episodes=args.n_iterations_val,
        n_way=args.n_way_val, n_shot=args.n_shot_val, n_query=args.n_query_val)
    val_loader = val_datamgr.get_data_loader(val_file, aug=False)

    print("##"*27, f"OBJECTIVE: {aux_func.__name__}", "##"*27)

    # Iteratively optimize some objective and evaluate performance
    trainer = Generic_adaptation_trainer(
        algorithm=algorithm,
        aux_objective=aux_func,
        outer_loss_func=loss_func,
        outer_optimizer=optimizer, 
        writer=writer,
        log_interval=args.log_interval, grad_clip=args.grad_clip,
        model_type=args.model_type,
        n_aux_objective_steps=args.n_aux_objective_steps)
    
    # print results
    results = trainer.run(val_loader, val_datamgr, is_training=False, start=0)
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(results)
        



if __name__ == '__main__':

    def str2bool(arg):
        return arg.lower() == 'true'

    parser = argparse.ArgumentParser(
        description='Analysis of Inner Solver methods for Meta Learning')

    # Model
    parser.add_argument('--model-type', type=str, default='resnet',
        help='type of the model')
    parser.add_argument('--distance-classifier', action='store_true', default=False,
        help='use a distance classifer (cosine based)')
    parser.add_argument('--num-classes', type=int, default=200,
        help='no of classes -- used during fine tuning')
    parser.add_argument('--label-offset', type=int, default=0,
        help='offset for label values during fine tuning stage')
    parser.add_argument('--no-fc-layer', type=str2bool, default=True,
        help='will not add fc layer to model')

    # Optimization
    parser.add_argument('--lr', type=float, default=0.001,
        help='learning rate for the global update')
    parser.add_argument('--grad-clip', type=float, default=0.0,
        help='gradient clipping')
    parser.add_argument('--n-epochs', type=int, default=60000,
        help='number of model training epochs')
    parser.add_argument('--n-aux-objective-steps', type=int, default=5,
        help='number of gradient updates on the auxiliary objective for each task')
    parser.add_argument('--add-bias', type=str2bool, default=False,
        help='add bias term inner loop')

    # Dataset
    parser.add_argument('--dataset-path', type=str,
        help='which dataset to use')
    parser.add_argument('--batch-size-train', type=int, default=10,
        help='batch size for training')
    parser.add_argument('--batch-size-val', type=int, default=10,
        help='batch size for validation')
    parser.add_argument('--n-query-train', type=int, default=15,
        help='how many samples per class for validation (meta train)')
    parser.add_argument('--n-query-val', type=int, default=15,
        help='how many samples per class for validation (meta val)')
    parser.add_argument('--n-shot-train', type=int, default=5,
        help='how many samples per class for train (meta train)')
    parser.add_argument('--n-shot-val', type=int, default=5,
        help='how many samples per class for train (meta val)')
    parser.add_argument('--n-way-train', type=int, default=5,
        help='how classes per task for train (meta train)')
    parser.add_argument('--n-way-val', type=int, default=5,
        help='how classes per task for train (meta val)')
    parser.add_argument('--img-side-len', type=int, default=28,
        help='width and height of the input images')
    

    # Miscellaneous
    parser.add_argument('--output-folder', type=str, default='maml',
        help='name of the output folder')
    parser.add_argument('--device-number', type=str, default='0',
        help='gpu device number')
    parser.add_argument('--log-interval', type=int, default=100,
        help='number of batches between tensorboard writes')
    parser.add_argument('--checkpoint', type=str, default='',
        help='path to saved parameters.')
    parser.add_argument('--train-aug', action='store_true', default=True,
        help='perform data augmentation during training')
    
    args = parser.parse_args()

    # Create logs and saves folder if they don't exist
    if not os.path.exists('./train_dir'):
        os.makedirs('./train_dir')

    # main function call
    main(args)
