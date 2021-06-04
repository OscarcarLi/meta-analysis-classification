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
import src.logger
import sys


from src.models import shallow_conv, resnet_12, wide_resnet, dense_net
from src.algorithm_trainer.algorithm_trainer import Meta_algorithm_trainer, Init_algorithm_trainer, TL_algorithm_trainer
from src.algorithms.algorithm import SVM, ProtoNet, Ridge, InitBasedAlgorithm
from src.data.fedlearn_datasets_fixedsq import FedDataset_Fix, FedDataLoader, SimpleFedDataset


def str2bool(arg):
    return arg.lower() == 'true'


def ensure_path(path):
    assert os.path.exists(path), "Output folder should match the one of the chkpt"
    return path



def main(args):


    ####################################################
    #                LOGGING AND SAVING                #
    ####################################################
    assert args.checkpoint != '' 
    # this script needs a pre-determined checkpoint to run eval
    args.output_folder = ensure_path('./runs/{0}'.format(args.output_folder))
    print(f'Will write eval results to {args.output_folder}')
    
    time_now = datetime.now(pytz.timezone("America/New_York")).strftime("%d:%b:%Y:%H:%M:%S")
    with open(f'{args.output_folder}/eval_config_{time_now}.txt', 'w') as config_txt:
        for k, v in sorted(vars(args).items()):
            config_txt.write(f'{k}: {v}\n')
    
    # replace stdout with Logger; the original sys.stdout is saved in src.logger.stdout
    sys.stdout = src.logger.Logger(log_filename=f'{args.output_folder}/eval_{time_now}.log')
    src.logger.stdout.write('hi!')

    # file to maintain a record of evaluations
    # each record in this file has format: checkpoint val-json-path (if present) novel-json-path (if present) 
    # followed by accStd 
    eval_results = f'{args.output_folder}/{args.eval_records_path}'
    eval_results_record = f'Chkpt:{args.checkpoint}'
    

    ####################################################
    #         DATASET AND DATALOADER CREATION          #
    ####################################################

    # json paths
    dataset_name = args.dataset_path.split('/')[-1]
    image_size = args.img_side_len
    val_file = os.path.join(args.dataset_path, args.val_json)
    test_file = os.path.join(args.dataset_path, args.novel_json)
    print("Dataset name", dataset_name, "image_size", image_size)
    
    # log to eval results file
    if args.val_json != '':
        eval_results_record += f' ValJson:{val_file}'
    if args.novel_json != '':
        eval_results_record += f' TestJson:{test_file}'

    
    """
    1. Create Val/Test FedDataset object, which handles preloading of images for every single client
    2. Create FedDataLoader object from each FedDataset, which samples a batch of clients.
    """

    if args.val_json != '':
        print("\n", "--"*20, "VAL", "--"*20) 
        val_dataset = FedDataset_Fix(
                        json_path=val_file,
                        image_size=(image_size, image_size),
                        preload=False,)
        val_loader = FedDataLoader(
                        dataset=val_dataset,
                        n_batches=0,
                        batch_size=args.batch_size_val)

    if args.novel_json != '':
        print("\n", "--"*20, "TEST", "--"*20)
        test_dataset = FedDataset_Fix(
                        json_path=test_file,
                        image_size=(image_size, image_size),
                        preload=False,)
        test_loader = FedDataLoader(
                        dataset=test_dataset,
                        n_batches=0,
                        batch_size=args.batch_size_val)

    
    ####################################################
    #             MODEL/BACKBONE CREATION              #
    ####################################################

    print("\n", "--"*20, "MODEL", "--"*20)

    if args.model_type == 'resnet_12':
        if 'miniimagenet' in dataset_name.lower() or 'cub' in dataset_name.lower() or 'celeba' in dataset_name.lower():
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
    #                LOAD FROM CHECKPOINT              #
    ####################################################

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
    else:
        raise ValueError(
            'Unrecognized algorithm {}'.format(args.algorithm))


    if args.algorithm == 'InitBasedAlgorithm':
        trainer = Init_algorithm_trainer(
            algorithm=algorithm,
            optimizer=None,
            writer=None,
            log_interval=args.log_interval, 
            save_folder=None, 
            grad_clip=None,
            num_updates_inner_train=args.num_updates_inner_train,
            num_updates_inner_val=args.num_updates_inner_val,
            init_global_iteration=None)
    else:        
        trainer = Meta_algorithm_trainer(
            algorithm=algorithm,
            optimizer=None,
            writer=None,
            log_interval=args.log_interval, 
            save_folder=None, 
            grad_clip=None,
            init_global_iteration=None)
        

    ####################################################
    #                      VAL/TEST                    #
    ####################################################

    if args.val_json != '':
        print("Validation")
        results = trainer.run(
            mt_loader=val_loader, is_training=False, 
            evaluate_supervised_classification=False)
        eval_results_record += f" ValResult: {results['val_task_acc']}"
        print(pprint.pformat(results, indent=4))

    if args.novel_json != '':
        print("Test")
        results = trainer.run(
            mt_loader=test_loader, is_training=False,
            evaluate_supervised_classification=False)
        eval_results_record += f" TestResult: {results['val_task_acc']}"
        print(pprint.pformat(results, indent=4))    

    # write record to eval records file
    with open(eval_results, 'a') as f:
        f.write(f'{eval_results_record}' + '\n')


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
    parser.add_argument('--scale-factor', type=float, default=1.,
        help='scalar factor multiplied with logits')
    parser.add_argument('--learnable-scale', type=str, default="False",
        help='scalar receives grads')

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
    parser.add_argument('--val-json', type=str, default='',
        help='val json name')
    parser.add_argument('--novel-json', type=str, default='',
        help='novel json name')
    parser.add_argument('--img-side-len', type=int, default=84,
        help='width and height of the input images')
    parser.add_argument('--num-classes-train', type=int, default=0,
        help='no of train classes')
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
    parser.add_argument('--classifier-metric', type=str, default='',
        help='')
    parser.add_argument('--projection', type=str, default='',
        help='')
    parser.add_argument('--avg-pool', type=str, default='True',
        help='')
    parser.add_argument('--val-frequency', type=int, default=1,
        help='') 
    parser.add_argument('--eval-records-path', type=str, default='eval_records.txt',
        help='path to file that stores eval records')
    
    
    args = parser.parse_args()

    # set random seed. only set for numpy, uncomment the below lines for torch and CuDNN.
    if args.random_seed != 0:
        np.random.seed(args.random_seed)
    
    # main function call
    main(args)