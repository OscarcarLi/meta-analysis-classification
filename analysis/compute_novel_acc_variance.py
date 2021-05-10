import os
from tqdm import tqdm
import json
import argparse
import pickle
import torch
import torch.nn as nn
import numpy as np
import pprint
import re
import shutil
from datetime import datetime


from src.models import shallow_conv, resnet_12, wide_resnet, dense_net
from src.algorithm_trainer.algorithm_trainer import Meta_algorithm_trainer, Init_algorithm_trainer, TL_algorithm_trainer
from src.algorithms.algorithm import SVM, ProtoNet, Ridge, InitBasedAlgorithm
from src.optimizers import modified_sgd
from src.data.dataset_managers import MetaDataLoader
from src.data.datasets import MetaDataset, ClassImagesSet, SimpleDataset


def ensure_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def str2bool(arg):
    return arg.lower() == 'true'


def create_model_and_load_chkpt(dataset_name, algorithm_hyperparams):
    
    print("\n", "--"*20, "MODEL", "--"*20)

    if algorithm_hyperparams['model_type'] == 'resnet_12':
        if 'miniImagenet' in dataset_name or 'CUB' in dataset_name:
            model = resnet_12.resnet12(
                avg_pool=str2bool(algorithm_hyperparams['avg_pool']),
                drop_rate=0.1, dropblock_size=5,
                num_classes=algorithm_hyperparams['num_classes_train'], 
                classifier_type=algorithm_hyperparams['classifier_type'],
                projection=str2bool(algorithm_hyperparams['projection']),
                learnable_scale=str2bool(algorithm_hyperparams['learnable_scale']))

        else:
            model = resnet_12.resnet12(
                avg_pool=str2bool(algorithm_hyperparams['avg_pool']),
                drop_rate=0.1, dropblock_size=2,
                num_classes=algorithm_hyperparams['num_classes_train'], 
                classifier_type=algorithm_hyperparams['classifier_type'],
                projection=str2bool(algorithm_hyperparams['projection']),
                learnable_scale=str2bool(algorithm_hyperparams['learnable_scale']))

    elif algorithm_hyperparams['model_type'] in ['conv64', 'conv48', 'conv32']:
        dim = int(args.model_type[-2:])
        model = shallow_conv.ShallowConv(
            z_dim=dim,
            h_dim=dim,
            num_classes=algorithm_hyperparams['num_classes_train'],
            x_width=args.img_side_len,
            classifier_type=algorithm_hyperparams['classifier_type'],
            projection=str2bool(algorithm_hyperparams['projection']),
            learnable_scale=str2bool(algorithm_hyperparams['learnable_scale']))

    elif algorithm_hyperparams['model_type'] == 'wide_resnet28_10':
        model = wide_resnet.wrn28_10(
            projection=str2bool(algorithm_hyperparams['projection']), 
            classifier_type=algorithm_hyperparams['classifier_type'],
            learnable_scale=str2bool(algorithm_hyperparams['learnable_scale']))

    elif algorithm_hyperparams['model_type'] == 'wide_resnet16_10':
        model = wide_resnet.wrn16_10(
            projection=str2bool(algorithm_hyperparams['projection']), 
            classifier_type=algorithm_hyperparams['classifier_type'],
            learnable_scale=str2bool(algorithm_hyperparams['learnable_scale']))

    else:
        raise ValueError(
            'Unrecognized model type {}'.format(args.model_type))

    print("Model\n" + "=="*27)    
    print(model)   

    print(f"loading model from {algorithm_hyperparams['checkpoint']}")
    model_dict = model.state_dict()
    chkpt = torch.load(
        algorithm_hyperparams['checkpoint'], 
        map_location=torch.device('cpu'))
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
        
    # Multi-gpu support and device setup
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_number
    print('Using GPUs: ', os.environ["CUDA_VISIBLE_DEVICES"])
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.cuda()

    return model



def create_alg_and_trainer(args, algorithm_hyperparams, model):

    # algorithm
    if algorithm_hyperparams['algorithm'] == 'InitBasedAlgorithm':
        algorithm = InitBasedAlgorithm(
            model=model,
            loss_func=torch.nn.CrossEntropyLoss(),
            method=algorithm_hyperparams['init_meta_algorithm'],
            alpha=algorithm_hyperparams['alpha'],
            inner_loop_grad_clip=algorithm_hyperparams['grad_clip_inner'],
            inner_update_method=algorithm_hyperparams['inner_update_method'],
            device='cuda')
    elif algorithm_hyperparams['algorithm'] == 'ProtoNet':
        algorithm = ProtoNet(
            model=model,
            inner_loss_func=torch.nn.CrossEntropyLoss(),
            device='cuda',
            scale=algorithm_hyperparams['scale_factor'],
            metric=algorithm_hyperparams['classifier_metric'])
    elif algorithm_hyperparams['algorithm'] == 'SVM':
        algorithm = SVM(
            model=model,
            inner_loss_func=torch.nn.CrossEntropyLoss(),
            scale=algorithm_hyperparams['scale_factor'],
            device='cuda')
    elif algorithm_hyperparams['algorithm'] == 'Ridge':
        algorithm = Ridge(
            model=model,
            inner_loss_func=torch.nn.CrossEntropyLoss(),
            scale=algorithm_hyperparams['scale_factor'],
            device='cuda')
    elif algorithm_hyperparams['algorithm'] == 'TransferLearning':
        """
        We use the ProtoNet algorithm at test time.
        """
        algorithm = ProtoNet(
            model=model,
            inner_loss_func=torch.nn.CrossEntropyLoss(),
            device='cuda',
            scale=algorithm_hyperparams['scale_factor'],
            metric=algorithm_hyperparams['classifier_metric'])
    else:
        raise ValueError(
            'Unrecognized algorithm {}'.format(algorithm_hyperparams['algorithm']))


    if algorithm_hyperparams['algorithm'] == 'InitBasedAlgorithm':
        trainer = Init_algorithm_trainer(
            algorithm=algorithm,
            optimizer=None,
            writer=None,
            log_interval=args.log_interval, 
            save_folder='', 
            grad_clip=None,
            num_updates_inner_train=algorithm_hyperparams['num_updates_inner_train'],
            num_updates_inner_val=algorithm_hyperparams['num_updates_inner_val'],
            init_global_iteration=None)
    elif algorithm_hyperparams['algorithm'] == 'TransferLearning':
        trainer = TL_algorithm_trainer(
            algorithm=algorithm,
            optimizer=None,
            writer=None,
            log_interval=args.log_interval, 
            save_folder='', 
            grad_clip=None,
            init_global_iteration=None
        )
    else:
        trainer = Meta_algorithm_trainer(
            algorithm=algorithm,
            optimizer=None,
            writer=None,
            log_interval=args.log_interval, 
            save_folder='', 
            grad_clip=None,
            init_global_iteration=None)

    return trainer


def main(args):

    ####################################################
    #                LOGGING AND SAVING                #
    ####################################################
    args.output_folder = ensure_path('./runs/{0}'.format(args.output_folder))
    eval_results = f'{args.output_folder}/novel_acc_variance_{args.n_chosen_classes}.txt'
    with open(eval_results, 'a') as f:
        f.write("--"*20 + "EVALUATION RESULTS" + "--"*20 + '\n')


    ####################################################
    #                LOAD ALG HYPERPARAMS              #
    ####################################################

    algorithm_hyperparams_1 = algorithm_hyperparams_2 = None
    with open(args.algorithm_hyperparams_json_1, 'r') as f:
        algorithm_hyperparams_1 = json.load(f)
    if args.algorithm_hyperparams_json_2 != '':
        with open(args.algorithm_hyperparams_json_2, 'r') as f:
            algorithm_hyperparams_2 = json.load(f)


    ####################################################
    #         DATASET AND DATALOADER CREATION          #
    ####################################################

    # json paths
    dataset_name = args.dataset_path.split('/')[-1]
    image_size = args.img_side_len
    # train_file = os.path.join(args.dataset_path, 'base_test.json')
    # val_file = os.path.join(args.dataset_path, 'val.json')
    test_file = os.path.join(args.dataset_path, 'novel_large.json')
    basetest_file = os.path.join(args.dataset_path, 'base_test.json')
    print("Dataset name", dataset_name, "image_size", image_size)
    
    """
    1. Create ClassImagesSet object, which handles preloading of images
    2. Pass ClassImagesSet to MetaDataset which handles nshot, nquery and fixSupport
    3. Create Dataloader object from MetaDataset
    """

    print("\n", "--"*20, "VAL + NOVEL", "--"*20)
    
    
    if  algorithm_hyperparams_1['algorithm'] == 'TransferLearning':
        if algorithm_hyperparams_2 is not None:
            assert algorithm_hyperparams_2['algorithm'] == 'TransferLearning'
        base_classes = ClassImagesSet(basetest_file, preload=str2bool(args.preload_images))
    else:
        novelval_classes = ClassImagesSet(test_file, preload=str2bool(args.preload_images))
        novelval_meta_datasets = MetaDataset(
                                    dataset_name=dataset_name,
                                    support_class_images_set=novelval_classes,
                                    query_class_images_set=novelval_classes,
                                    image_size=image_size,
                                    support_aug=False,
                                    query_aug=False,
                                    fix_support=0,
                                    save_folder='')


    ####################################################
    #             MODEL/BACKBONE CREATION              #
    ####################################################
    
    model_1 = create_model_and_load_chkpt(
                    dataset_name=dataset_name, 
                    algorithm_hyperparams=algorithm_hyperparams_1)
    if algorithm_hyperparams_2 is not None:
        model_2 = create_model_and_load_chkpt(
                        dataset_name=dataset_name, 
                        algorithm_hyperparams=algorithm_hyperparams_2)

    ####################################################
    #        ALGORITHM AND ALGORITHM TRAINER           #
    ####################################################

    trainer_1 = create_alg_and_trainer(
                    args, 
                    algorithm_hyperparams=algorithm_hyperparams_1,
                    model=model_1)
    if algorithm_hyperparams_2 is not None:
        trainer_2 = create_alg_and_trainer(
                        args, 
                        algorithm_hyperparams=algorithm_hyperparams_2,
                        model=model_2)

    ####################################################
    #                    EVALUATION                    #
    ####################################################

    # check if there is pre specified list of novel classes for each run
    chosen_classes_indices_list = None
    if args.chosen_classes_indices_list != '':
        print(f'Loading novel classes for each run from file {args.chosen_classes_indices_list}') 
        chosen_classes_indices_list = []
        with open(args.chosen_classes_indices_list, 'r') as f:
            chosen_classes_indices_list = [line.strip().split(" ") for line in f.readlines()]
    
    if algorithm_hyperparams_1['algorithm'] == 'TransferLearning':
        # compare two supervised learning models
        assert algorithm_hyperparams_2 is None or algorithm_hyperparams_2['algorithm'] == 'TransferLearning' 
        with open(eval_results, 'a') as f:
            f.write('f{dataset_name} {args.sample} examples sampled from each of {len(novelval_classes)} class\n')
    else:
        # compare meta-learning algorithm snapshots
        with open(eval_results, 'a') as f:
            f.write(f'{dataset_name} {args.n_chosen_classes} out of {len(novelval_classes)} classes\n')
            f.write(f"Alg1: {algorithm_hyperparams_1['checkpoint']}")
            if algorithm_hyperparams_2 is not None:
                f.write(f", Alg2: {algorithm_hyperparams_2['checkpoint']}")
            f.write('\n')


    for run in range(args.n_runs):
        print(f'{run+1} out of {args.n_runs}')
        if algorithm_hyperparams_1['algorithm'] == 'TransferLearning':

            novelval_dataset = SimpleDataset(
                            dataset_name=dataset_name,
                            class_images_set=base_classes,
                            image_size=image_size,
                            aug=False,
                            sample=args.sample) # fix and only sample args.sample number of examples for each class.
            novelval_loaders = torch.utils.data.DataLoader(
                            novelval_dataset, 
                            batch_size=128, 
                            shuffle=True,
                            num_workers=6)

            results_1 = trainer_1.run(
                mt_loader=novelval_loaders, is_training=False, evaluate_supervised_classification=True)
            if algorithm_hyperparams_2 is not None:
                results_2 = trainer_2.run(
                    mt_loader=novelval_loaders, is_training=False, evaluate_supervised_classification=True)
            
            with open(eval_results, 'a') as f:
                f.write(f"Run{run+1}: ")
                f.write(f"Alg_1: Loss {round(results_1['test_loss_after']['loss'], 3)} Acc {round(results_1['test_loss_after']['accu'], 3)} ")
                if algorithm_hyperparams_2 is not None:
                    f.write(f"Alg_2: Loss {round(results_2['test_loss_after']['loss'], 3)} Acc {round(results_2['test_loss_after']['accu'], 3)}")            
                f.write("\n")

        else:
            # evaluation on meta-learning algorithm 
            if chosen_classes_indices_list is None:
                chosen_classes = np.random.choice(
                    list(novelval_classes.keys()), args.n_chosen_classes,replace=False)
            else:
                novel_class_keys = sorted(list(novelval_classes.keys())) # use sorted to ensure consistency over multiple runs
                chosen_classes = [novel_class_keys[int(idx)] for idx in chosen_classes_indices_list[run]] # take the chosen_classes for this run.
            
            # when the chosen classes is read from the saved txt file
            # need to make sure we have the correct number classes for each eval
            assert args.n_chosen_classes == len(chosen_classes)
            novelval_loaders = MetaDataLoader(
                dataset=novelval_meta_datasets,
                n_batches=args.n_iterations_val,
                batch_size=args.batch_size_val,
                n_way=args.n_way_val,
                n_shot=args.n_shot_val,
                n_query=args.n_query_val, 
                randomize_query=False,
                verbose=False,
                p_dict={
                    k: (1 / args.n_chosen_classes if k in chosen_classes else 0.)
                        for k in list(novelval_classes)
                })

            results_1 = trainer_1.run(
                mt_loader=novelval_loaders, is_training=False, verbose=False)
            if algorithm_hyperparams_2 is not None:
                results_2 = trainer_2.run(
                    mt_loader=novelval_loaders, is_training=False, verbose=False)
        
        
            with open(eval_results, 'a') as f:
                f.write(f"Run{run+1} {args.n_way_val}w{args.n_shot_val}s: ")
                if len(chosen_classes) != len(novelval_classes):
                    f.write(f"Classes [{' '.join([str(x) for x in sorted(chosen_classes)])}] ")
                f.write(f"Alg1: Loss {round(results_1['test_loss_after']['loss'], 3)} Acc {round(results_1['test_loss_after']['accu'], 3)} AccStd {results_1['val_task_acc']}")
                if algorithm_hyperparams_2 is not None:
                    f.write(f" Alg2: Loss {round(results_2['test_loss_after']['loss'], 3)} Acc {round(results_2['test_loss_after']['accu'], 3)} AccStd {results_2['val_task_acc']}")
                f.write("\n")



if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Training the feature backbone on all classes from all tasks.')
    
    parser.add_argument('--random-seed', type=int, default=0,
        help='')

    # algorithm hyperparams
    parser.add_argument('--algorithm-hyperparams-json-1', type=str, default='',
        help='path to json with hyper parameters for alg1.')
    parser.add_argument('--algorithm-hyperparams-json-2', type=str, default='',
        help='path to json with hyper parameters for alg2.')

    # Evaluation specific parameters
    parser.add_argument('--n-chosen-classes', type=int, default=5,
        help='number of classes chosen for eval in a single run')
    parser.add_argument('--n-runs', type=int, default=20,
        help='number of runs')
    parser.add_argument('--sample', type=int, default=0,
        help='samples per class; for SimpleDataset Supervised Learning')
    parser.add_argument('--chosen-classes-indices-list', type=str, default='',
        help='a txt file with each row specifying which 0-based class indices to choose for evaluation')

    # Dataset
    parser.add_argument('--dataset-path', type=str,
        help='which dataset to use')
    parser.add_argument('--img-side-len', type=int, default=84,
        help='width and height of the input images')
    parser.add_argument('--batch-size-val', type=int, default=10,
        help='batch size for validation')
    parser.add_argument('--n-query-val', type=int, default=15,
        help='how many samples per class for validation (meta val)')
    parser.add_argument('--n-shot-val', type=int, default=5,
        help='how many samples per class for train (meta val)')
    parser.add_argument('--n-way-val', type=int, default=5,
        help='how classes per task for train (meta val)')
    parser.add_argument('--n-iterations-val', type=int, default=100,
        help='no. of iterations validation.') 
    parser.add_argument('--preload-images', type=str, default="True")
    parser.add_argument('--num-classes-train', type=int, default=0,
        help='no of train classes')
    parser.add_argument('--num-classes-val', type=int, default=0,
        help='no of novel (val) classes')
    
    # Miscellaneous
    parser.add_argument('--output-folder', type=str, default='maml',
        help='name of the output folder')
    parser.add_argument('--device-number', type=str, default='0',
        help='gpu device number')
    parser.add_argument('--log-interval', type=int, default=100,
        help='number of batches between tensorboard writes')

    # Parse Args
    args = parser.parse_args()

    # set random seed. only set for numpy, uncomment the below lines for torch and CuDNN.
    if args.random_seed != 0:
        np.random.seed(args.random_seed)
    
    # main function call
    main(args)
