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
import pytz


from src.models import shallow_conv, resnet_12, wide_resnet, dense_net
from src.algorithm_trainer.algorithm_trainer import Meta_algorithm_trainer, Init_algorithm_trainer, TL_algorithm_trainer
from src.algorithms.algorithm import SVM, ProtoNet, Ridge, InitBasedAlgorithm
from src.optimizers import modified_sgd
# from src.data.dataset_managers import MetaDataLoader
# from src.data.datasets import MetaDataset, ClassImagesSet, SimpleDataset

from src.data.fedlearn_datasets_fixedsq import FedDataset_Fix, FedDataLoader, SimpleFedDataset

import src.logger
import sys


def ensure_path(path):
    if os.path.exists(path):
        print("Path Exists", path, "Appending timestamp")
        timezone = pytz.timezone("America/New_York")
        path = path + "_" + datetime.now(timezone).strftime("%d:%b:%Y:%H:%M:%S")
        print("New path is", path)
        os.makedirs(path)
    else:
        os.makedirs(path)
    return path


def str2bool(arg):
    return arg.lower() == 'true'


def main(args):


    ####################################################
    #                LOGGING AND SAVING                #
    ####################################################
    if args.checkpoint != '':
        # if we are reloading, we don't need to timestamp and create a new folder
        # instead keep writing to the original output_folder
        assert os.path.exists(f'./runs/{args.output_folder}')
        args.output_folder = f'./runs/{args.output_folder}'
        print(f'resume training and will write to {args.output_folder}')
    else:
        args.output_folder = ensure_path('./runs/{0}'.format(args.output_folder))
    writer = SummaryWriter(args.output_folder)
    time_now = datetime.now(pytz.timezone("America/New_York")).strftime("%d:%b:%Y:%H:%M:%S")
    with open(f'{args.output_folder}/config_{time_now}.txt', 'w') as config_txt:
        for k, v in sorted(vars(args).items()):
            config_txt.write(f'{k}: {v}\n')
    save_folder = args.output_folder

    # replace stdout with Logger; the original sys.stdout is saved in src.logger.stdout
    sys.stdout = src.logger.Logger(log_filename=f'{args.output_folder}/train_{time_now}.log')
    src.logger.stdout.write('hi!')

    ####################################################
    #         DATASET AND DATALOADER CREATION          #
    ####################################################

    # json paths
    dataset_name = args.dataset_path.split('/')[-1]
    image_size = args.img_side_len
    train_file = os.path.join(args.dataset_path, 'base.json')
    val_file = os.path.join(args.dataset_path, 'val.json')
    test_file = os.path.join(args.dataset_path, 'novel.json')
    print("Dataset name", dataset_name, "image_size", image_size)

    
    """
    1. Create FedDataset object, which handles preloading of images for every single client
    2. Create FedDataLoader object from FedDataset, which samples a batch of clients.
    """

    print("\n", "--"*20, "TRAIN", "--"*20)
    if args.algorithm in ["SupervisedBaseline", "TransferLearning"]:
        """
        For Transfer Learning we create a SimpleFedDataset.
        """
        
        train_dataset = SimpleFedDataset(
                            json_path=train_file,
                            image_size=(image_size, image_size), # has to be a (h, w) tuple
                            preload=str2bool(args.preload_train))

        train_loader = torch.utils.data.DataLoader(
                            train_dataset, 
                            batch_size=args.batch_size_train, 
                            shuffle=True,
                            num_workers=6)
    else:
        train_meta_dataset = FedDataset_Fix(
                                json_path=train_file,
                                image_size=(image_size, image_size), # has to be a (h, w) tuple
                                preload=str2bool(args.preload_train),)
                               

        train_loader = FedDataLoader(
                            dataset=train_meta_dataset,
                            n_batches=0, # n_batches=0 means cycle sampling with random permutation through the dataset once
                            batch_size=args.batch_size_train)

    print("\n", "--"*20, "VAL", "--"*20) 
    if args.algorithm in ["SupervisedBaseline", "TransferLearning"]:
        val_dataset = SimpleFedDataset(
                            json_path=val_file,
                            image_size=(image_size, image_size), # has to be a (h, w) tuple
                            preload=True)

        val_loader = torch.utils.data.DataLoader(
                            val_dataset, 
                            batch_size=args.batch_size_val, 
                            shuffle=False,
                            drop_last=False,
                            num_workers=6)
    else:
        val_meta_dataset= FedDataset_Fix(
                                json_path=val_file,
                                image_size=(image_size, image_size),
                                preload=False,)
        val_loader = FedDataLoader(
                            dataset=val_meta_dataset,
                            n_batches=0,
                            batch_size=args.batch_size_val)

    print("\n", "--"*20, "TEST", "--"*20)
    if args.algorithm in ["SupervisedBaseline", "TransferLearning"]:
        test_dataset = SimpleFedDataset(
                            json_path=test_file,
                            image_size=(image_size, image_size), # has to be a (h, w) tuple
                            preload=True)

        test_loader = torch.utils.data.DataLoader(
                            test_dataset, 
                            batch_size=args.batch_size_val, 
                            shuffle=False,
                            drop_last=False,
                            num_workers=6)
    else:
        test_meta_dataset = FedDataset_Fix(
                                json_path=test_file,
                                image_size=(image_size, image_size),
                                preload=False,)
        test_loader = FedDataLoader(
                            dataset=test_meta_dataset,
                            n_batches=0,
                            batch_size=args.batch_size_val)

    
    ####################################################
    #             MODEL/BACKBONE CREATION              #
    ####################################################

    print("\n", "--"*20, "MODEL", "--"*20)

    if args.model_type == 'resnet_12':
        if 'miniImagenet' in dataset_name or 'CUB' in dataset_name or 'celeba' in dataset_name:
            model = resnet_12.resnet12(
                        avg_pool=str2bool(args.avg_pool),
                        drop_rate=0.1,
                        dropblock_size=5,
                        num_classes=args.num_classes_train,
                        classifier_type=args.classifier_type,
                        projection=str2bool(args.projection),
                        learnable_scale=str2bool(args.learnable_scale))
        else:
            model = resnet_12.resnet12(
                        avg_pool=str2bool(args.avg_pool),
                        drop_rate=0.1,
                        dropblock_size=2,
                        num_classes=args.num_classes_train,
                        classifier_type=args.classifier_type,
                        projection=str2bool(args.projection),
                        learnable_scale=str2bool(args.learnable_scale))

    elif args.model_type in ['conv64', 'conv48', 'conv32']:
        dim = int(args.model_type[-2:])
        model = shallow_conv.ShallowConv(
                    z_dim=dim,
                    h_dim=dim,
                    num_classes=args.num_classes_train, 
                    classifier_type=args.classifier_type,
                    projection=str2bool(args.projection),
                    learnable_scale=str2bool(args.learnable_scale))
    elif args.model_type == 'wide_resnet28_10':
        model = wide_resnet.wrn28_10(
                    projection=str2bool(args.projection),
                    classifier_type=args.classifier_type,
                    learnable_scale=str2bool(args.learnable_scale))
    elif args.model_type == 'wide_resnet16_10':
        model = wide_resnet.wrn16_10(
                    projection=str2bool(args.projection), 
                    classifier_type=args.classifier_type,
                    learnable_scale=str2bool(args.learnable_scale))
    else:
        raise ValueError(
            'Unrecognized model type {}'.format(args.model_type))
    print("Model\n" + "=="*27)    
    print(model)   



    ####################################################
    #                OPTIMIZER CREATION                #
    ####################################################

    # optimizer construction
    print("\n", "--"*20, "OPTIMIZER", "--"*20)
    print("Optimzer", args.optimizer_type)
    if args.optimizer_type.lower() == 'adam':
        optimizer = torch.optim.Adam([
            {'params': model.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay}
        ])
    else:
        optimizer = modified_sgd.SGD([
            {'params': model.parameters(), 'lr': args.lr,
            'weight_decay': args.weight_decay, 'momentum': 0.9, 'nesterov': True},
        ])
    print("Total n_epochs: ", args.n_epochs)   
    
    # learning rate scheduler creation
    if args.lr_scheduler_type == 'deterministic':
        drop_eps = [int(x) for x in args.drop_lr_epoch.split(',')]
        if args.drop_factors != '':
            drop_factors = [float(x) for x in args.drop_factors.split(',')]
        else:
            drop_factors = [0.06, 0.012, 0.0024]

        print("Drop lr at epochs", drop_eps)
        print("Drop factors", drop_factors[:len(drop_eps)])

        assert len(drop_factors) >= len(drop_eps), "No enough drop factors"
        
        def lr_lambda(x):
            '''
            x is an epoch number
            drop_eps is assumed to an list of strictly increasing epoch numbers
            here we require len(drop_factors) >= len(drop_eps)
            ideally they are of the same length
            but technically the code can just not use the additional factors
            '''
            for i in range(len(drop_eps)):
                if x >= drop_eps[i]:
                    continue
                else:
                    if i == 0:
                        return 1.0
                    else:
                        return drop_factors[i-1]
            return drop_factors[len(drop_eps) - 1]

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lr_lambda, last_epoch=-1)
        for _ in range(args.restart_iter):
            lr_scheduler.step()

    elif args.lr_scheduler_type == 'val_based':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
            mode='max', patience=5, factor=0.1, min_lr=5e-6, threshold=0.5)    

    else:
        raise ValueError("Unimplemented lr scheduler")

    print("LR scheduler ", args.lr_scheduler_type)  


    ####################################################
    #                LOAD FROM CHECKPOINT              #
    ####################################################

    if args.checkpoint != '':
        print(f"loading model from {args.checkpoint}")
        model_dict = model.state_dict() # new model's state dict
        chkpt = torch.load(args.checkpoint, map_location=torch.device('cpu'))

        ### load model
        chkpt_state_dict = chkpt['model']
        chkpt_state_dict_old_keys = list(chkpt_state_dict.keys())
        # remove "module." from key, possibly present as it was dumped by data-parallel
        for key in chkpt_state_dict_old_keys:
            if 'module.' in key:
                new_key = re.sub('module\.', '',  key)
                chkpt_state_dict[new_key] = chkpt_state_dict.pop(key)
        load_model_state_dict = {k: v for k, v in chkpt_state_dict.items() if k in model_dict}
        model_dict.update(load_model_state_dict)
        # updated_keys = set(model_dict).intersection(set(chkpt_state_dict))
        print(f"Updated {len(load_model_state_dict.keys())} keys using chkpt")
        print("Following keys updated :", "\n".join(sorted(load_model_state_dict.keys())))
        missed_keys = set(model_dict).difference(set(load_model_state_dict))
        print(f"Missed {len(missed_keys)} keys")
        print("Following keys missed :", "\n".join(sorted(missed_keys)))
        model.load_state_dict(model_dict)

        ### load optimizer
        try:
            print(f"loading optimizer from {args.checkpoint}")
            optimizer.load_state_dict(chkpt['optimizer'].state_dict())
            print("Successfully loaded optimizer")

        except:
            print("Failed to load optimizer")
                    
        
    # Multi-gpu support and device setup
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_number
    print('Using GPUs: ', os.environ["CUDA_VISIBLE_DEVICES"])
    # move model to cuda
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.cuda()
    print("Successfully moved the model to cuda")

    ####################################################
    #        ALGORITHM AND ALGORITHM TRAINER           #
    ####################################################

    # start tboard from restart iter
    init_global_iteration = 0
    if args.restart_iter:
        init_global_iteration = args.restart_iter * len(train_dataset) 

    # algorithm
    if args.algorithm == 'InitBasedAlgorithm':
        algorithm = InitBasedAlgorithm(
            model=model,
            loss_func=torch.nn.CrossEntropyLoss(),
            method=args.init_meta_algorithm,
            alpha=args.alpha,
            inner_loop_grad_clip=args.grad_clip_inner,
            inner_update_method=args.inner_update_method,
            device='cuda')
    elif args.algorithm == 'ProtoNet':
        algorithm = ProtoNet(
            model=model,
            inner_loss_func=torch.nn.CrossEntropyLoss(),
            device='cuda',
            scale=args.scale_factor,
            metric=args.classifier_metric)
    elif args.algorithm == 'SVM':
        algorithm = SVM(
            model=model,
            inner_loss_func=torch.nn.CrossEntropyLoss(),
            scale=args.scale_factor,
            device='cuda')
    elif args.algorithm == 'Ridge':
        algorithm = Ridge(
            model=model,
            inner_loss_func=torch.nn.CrossEntropyLoss(),
            scale=args.scale_factor,
            device='cuda')
    elif args.algorithm in ['TransferLearning', 'SupervisedBaseline']:
        """
        We use the ProtoNet algorithm at test time.
        """
        algorithm = ProtoNet(
            model=model,
            inner_loss_func=torch.nn.CrossEntropyLoss(),
            device='cuda',
            scale=args.scale_factor,
            metric=args.classifier_metric)
    else:
        raise ValueError(
            'Unrecognized algorithm {}'.format(args.algorithm))


    if args.algorithm == 'InitBasedAlgorithm':
        trainer = Init_algorithm_trainer(
            algorithm=algorithm,
            optimizer=optimizer,
            writer=writer,
            log_interval=args.log_interval, 
            save_folder=save_folder, 
            grad_clip=args.grad_clip,
            num_updates_inner_train=args.num_updates_inner_train,
            num_updates_inner_val=args.num_updates_inner_val,
            init_global_iteration=init_global_iteration)
    elif args.algorithm in ['TransferLearning', 'SupervisedBaseline']:
        trainer = TL_algorithm_trainer(
            algorithm=algorithm,
            optimizer=optimizer,
            writer=writer,
            log_interval=args.log_interval, 
            save_folder=save_folder, 
            grad_clip=args.grad_clip,
            init_global_iteration=init_global_iteration
        )
    else:        
        trainer = Meta_algorithm_trainer(
            algorithm=algorithm,
            optimizer=optimizer,
            writer=writer,
            log_interval=args.log_interval, 
            save_folder=save_folder, 
            grad_clip=args.grad_clip,
            init_global_iteration=init_global_iteration)
        

    ####################################################
    #                  TRAINER LOOP                    #
    ####################################################

    print("\n", "--"*20, "BEGIN TRAINING", "--"*20)
    
    # iterate over training epochs
    for iter_start in range(args.restart_iter, args.n_epochs):

        # do SB type evaluation or not
        evaluate_SB = args.algorithm in ["SupervisedBaseline"]
        
        # training
        for param_group in optimizer.param_groups:
            print('\n\nlearning rate:', param_group['lr'])

        trainer.run(
            mt_loader=train_loader,
            is_training=True,
            epoch=iter_start + 1,
            evaluate_supervised_classification=evaluate_SB)

        if iter_start % args.val_frequency == 0:
            # validation/test
            print("Validation")
            results = trainer.run(
                mt_loader=val_loader, is_training=False, 
                evaluate_supervised_classification=evaluate_SB)
            print(pprint.pformat(results, indent=4))
            writer.add_scalar(
                f"val_acc", results['test_loss_after']['accu'], iter_start + 1)
            writer.add_scalar(
                f"val_loss", results['test_loss_after']['loss'], iter_start + 1)
            val_accu = results['test_loss_after']['accu']

            print("Test")
            results = trainer.run(
                mt_loader=test_loader, is_training=False,
                evaluate_supervised_classification=evaluate_SB)
            print(pprint.pformat(results, indent=4))
            writer.add_scalar(
                f"test_acc", results['test_loss_after']['accu'], iter_start + 1)
            writer.add_scalar(
                f"test_loss", results['test_loss_after']['loss'], iter_start + 1)
            

        # scheduler step
        if args.lr_scheduler_type == 'val_based':
            assert args.val_frequency == 1, "eval after every epoch is mandatory for val based lr scheduler"
            lr_scheduler.step(val_accu)
        else:
            lr_scheduler.step()


if __name__ == '__main__':

    
    parser = argparse.ArgumentParser(
        description='Training the feature backbone on all classes from all tasks.')
    
    parser.add_argument('--random-seed', type=int, default=0,
        help='')

    # Algorithm
    parser.add_argument('--algorithm', type=str, help='type of algorithm')


    # Model
    parser.add_argument('--model-type', type=str, default='resnet_12',
        help='type of the model')
    parser.add_argument('--classifier-type', type=str, default='no-classifier',
        help='classifier type [distance based, linear, GDA]')
    parser.add_argument('--loss-names', type=str, nargs='+', default='cross_ent',
        help='names of various loss functions that are part fo overall objective')
    parser.add_argument('--scale-factor', type=float, default=1.,
        help='scalar factor multiplied with logits')
    parser.add_argument('--learnable-scale', type=str, default="False",
        help='scalar receives grads')

    # Optimization
    parser.add_argument('--optimizer-type', type=str, default='SGDM',
        help='SGDM/Adam')
    parser.add_argument('--lr', type=float, default=0.001,
        help='learning rate for the global update')
    parser.add_argument('--grad-clip', type=float, default=0.0,
        help='gradient clipping')
    parser.add_argument('--n-epochs', type=int, default=60000,
        help='number of model training epochs')
    parser.add_argument('--weight-decay', type=float, default=0.,
        help='weight decay')
    parser.add_argument('--drop-lr-epoch', type=str, default='20,40,50')
    parser.add_argument('--drop-factors', type=str, default='')
    parser.add_argument('--lr-scheduler-type', type=str, default='deterministic')


    # Initialization-based methods
    parser.add_argument('--alpha', type=float, default=0.0,
        help='inner learning rate for init based methods')
    parser.add_argument('--init-meta-algorithm', type=str, default='MAML',
        help='MAML/Reptile/FOMAML')
    parser.add_argument('--grad-clip-inner', type=float, default=0.0,
        help='gradient clip value in inner loop')
    parser.add_argument('--num-updates-inner-train', type=int, default=1,
        help='number of updates in inner loop')
    parser.add_argument('--num-updates-inner-val', type=int, default=1,
        help='number of updates in inner loop val')
    parser.add_argument('--inner-update-method', type=str, default='sgd',
        help='inner update method can be sgd or adam')


    # Dataset 
    parser.add_argument('--dataset-path', type=str,
        help='which dataset to use')
    parser.add_argument('--img-side-len', type=int, default=84,
        help='width and height of the input images')
    parser.add_argument('--num-classes-train', type=int, default=0,
        help='no of train classes')
    parser.add_argument('--batch-size-train', type=int, default=20,
        help='batch size for training')
    parser.add_argument('--batch-size-val', type=int, default=10,
        help='batch size for validation')
    parser.add_argument('--eps', type=float, default=0.0,
        help='epsilon of label smoothing')
    parser.add_argument('--preload-train', type=str, default="False")
    

    # Miscellaneous
    parser.add_argument('--output-folder', type=str, default='maml',
        help='name of the output folder')
    parser.add_argument('--device-number', type=str, default='0',
        help='gpu device number')
    parser.add_argument('--log-interval', type=int, default=100,
        help='number of batches between tensorboard writes')
    parser.add_argument('--checkpoint', type=str, default='',
        help='path to saved parameters.')
    parser.add_argument('--restart-iter', type=int, default=0,
        help='iteration at restart') 
    parser.add_argument('--classifier-metric', type=str, default='',
        help='')
    parser.add_argument('--projection', type=str, default='',
        help='')
    parser.add_argument('--avg-pool', type=str, default='True',
        help='')
    parser.add_argument('--val-frequency', type=int, default=1,
        help='') 
    
    
    args = parser.parse_args()

    # set random seed. only set for numpy, uncomment the below lines for torch and CuDNN.
    if args.random_seed != 0:
        np.random.seed(args.random_seed)
    
    # main function call
    main(args)
