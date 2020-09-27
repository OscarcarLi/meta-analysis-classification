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


from algorithm_trainer.models import gated_conv_net_original, resnet, resnet_2, conv64, resnet_12
from algorithm_trainer.algorithm_trainer import Classical_algorithm_trainer, Generic_adaptation_trainer, MetaClassical_algorithm_trainer
from algorithm_trainer.algorithms.algorithm import SVM, ProtoNet, Finetune, ProtoCosineNet, Protomax, Ridge
from algorithm_trainer.utils import optimizer_to_device
from algorithm_trainer.algorithms import modified_sgd
from data_layer.dataset_managers import ClassicalDataManager, MetaDataManager
from analysis.objectives import var_reduction_disc, var_reduction_ortho, rfc_and_pc


def ensure_path(path):
    if os.path.exists(path):
        print("Deleting", path)
        # if input('{} exists, remove? ([y]/n)'.format(path)) != 'n':
        shutil.rmtree(path)
        os.makedirs(path)
    else:
        os.makedirs(path)


def main(args):
    is_training = not args.eval
    run_name = 'train' if is_training else 'eval'

    if is_training:
        ensure_path('./train_dir_2/{0}'.format(args.output_folder))
        writer = SummaryWriter('./train_dir_2/{0}/{1}'.format(
            args.output_folder, run_name))
        with open('./train_dir_2/{}/config.txt'.format(
            args.output_folder), 'w') as config_txt:
            for k, v in sorted(vars(args).items()):
                config_txt.write('{}: {}\n'.format(k, v))
    else:
        writer = None

    save_folder = './train_dir_2/{0}'.format(args.output_folder)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)


    ####################################################
    #         DATASET AND DATALOADER CREATION          #
    ####################################################

    # There are 3 tyoes of files: base, val, novel
    # Here we train on base and validate on val
    image_size = args.img_side_len
    train_file = os.path.join(args.dataset_path, 'base.json')
    mt_datamgr = MetaDataManager(
        image_size, batch_size=args.batch_size_train, n_episodes=1000,
        n_way=args.n_way_train, n_shot=args.n_shot_train, n_query=max(args.n_query_train, args.n_query_pool), fix_support=args.fix_support)
    mt_loader = mt_datamgr.get_data_loader(
        args.dataset_path.split('/')[-1], train_file, support_aug=args.support_aug, query_aug=args.train_aug)

    val_file = os.path.join(args.dataset_path, 'val.json')
    mt_val_datamgr = MetaDataManager(
        image_size, batch_size=args.batch_size_val, n_episodes=args.n_iterations_val,
        n_way=args.n_way_val, n_shot=args.n_shot_val, n_query=args.n_query_val)
    mt_val_loader = mt_val_datamgr.get_data_loader(
        args.dataset_path.split('/')[-1], val_file, support_aug=False, query_aug=False)
    
    test_file = os.path.join(args.dataset_path, 'novel.json')
    mt_test_datamgr = MetaDataManager(
        image_size, batch_size=args.batch_size_val, n_episodes=args.n_iterations_val,
        n_way=args.n_way_val, n_shot=args.n_shot_val, n_query=args.n_query_val)
    mt_test_loader = mt_test_datamgr.get_data_loader(
        args.dataset_path.split('/')[-1], test_file, support_aug=False, query_aug=False)
    

    ####################################################
    #             MODEL/BACKBONE CREATION              #
    ####################################################

    if args.model_type == 'resnet':
        model = resnet_2.ResNet18(num_classes=args.num_classes_train, 
            classifier_type=None, no_fc_layer=True)
    elif args.model_type == 'resnet12':
        if args.dataset_path.split('/')[-1] in ['miniImagenet', 'CUB']:
            model = resnet_12.resnet12(avg_pool=(args.avg_pool == "True"), drop_rate=0.1, dropblock_size=5,
                num_classes=args.num_classes_train, classifier_type=args.classifier_type,
                projection=(args.projection=="True"), classifier_metric=args.classifier_metric)
        else:
            model = resnet_12.resnet12(avg_pool=(args.avg_pool == "True"), drop_rate=0.1, dropblock_size=2,
                num_classes=args.num_classes_train, classifier_type=args.classifier_type,
                projection=(args.projection=="True"), classifier_metric=args.classifier_metric)
    elif args.model_type == 'conv64':
        model = conv64.Conv64(num_classes=args.num_classes_train, 
                classifier_type=None, no_fc_layer=True, projection=(args.projection=="True"), classifier_metric=args.classifier_metric)
    else:
        raise ValueError(
            'Unrecognized model type {}'.format(args.model_type))
    print("Model\n" + "=="*27)    
    print(model)   


    ####################################################
    #                OPTIMIZER CREATION                #
    ####################################################

    # optimizer = torch.optim.Adam([
    #     {'params': model.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay}
    # ])
    optimizer = modified_sgd.SGD([
        {'params': model.parameters(), 'lr': args.lr,
         'weight_decay': args.weight_decay, 'momentum': 0.9, 'nesterov': True},
    ])
    print("Total n_epochs: ", args.n_epochs)     


    # load from checkpoint
    if args.checkpoint != '':
        print(f"loading from {args.checkpoint}")
        model_dict = model.state_dict()
        chkpt_state_dict = torch.load(args.checkpoint)
        if 'model' in chkpt_state_dict:
            if 'optimizer' in chkpt_state_dict and args.load_optimizer:
                print(f"loading optimizer from {args.checkpoint}")
                optimizer.state = chkpt_state_dict['optimizer'].state
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
        

    ####################################################
    #                     TRAINER                      #
    ####################################################

    # algorithm
    if args.algorithm == 'ProtonetCosine':
        algorithm_train = ProtoCosineNet(
            model=model,
            inner_loss_func=torch.nn.CrossEntropyLoss(),
            n_way=args.n_way_train,
            n_shot=args.n_shot_train,
            n_query=args.n_query_train,
            device='cuda')
        algorithm_val = ProtoCosineNet(
            model=model,
            inner_loss_func=torch.nn.CrossEntropyLoss(),
            n_way=args.n_way_val,
            n_shot=args.n_shot_val,
            n_query=args.n_query_val,
            device='cuda')
    elif args.algorithm == 'Protonet':
        algorithm_train = ProtoNet(
            model=model,
            inner_loss_func=torch.nn.CrossEntropyLoss(),
            n_way=args.n_way_train,
            n_shot=args.n_shot_train,
            n_query=args.n_query_train,
            device='cuda')
        algorithm_val = ProtoNet(
            model=model,
            inner_loss_func=torch.nn.CrossEntropyLoss(),
            n_way=args.n_way_val,
            n_shot=args.n_shot_val,
            n_query=args.n_query_val,
            device='cuda')
    elif args.algorithm == 'Protomax':
        algorithm_train = Protomax(
            model=model,
            inner_loss_func=torch.nn.CrossEntropyLoss(),
            n_way=args.n_way_train,
            n_shot=args.n_shot_train,
            n_query=args.n_query_train,
            device='cuda')
        algorithm_val = ProtoNet(
            model=model,
            inner_loss_func=torch.nn.CrossEntropyLoss(),
            n_way=args.n_way_val,
            n_shot=args.n_shot_val,
            n_query=args.n_query_val,
            device='cuda')
    elif args.algorithm == 'SVM':
        algorithm_train = SVM(
            model=model,
            inner_loss_func=torch.nn.CrossEntropyLoss(),
            n_way=args.n_way_train,
            n_shot=args.n_shot_train,
            n_query=args.n_query_train,
            device='cuda')
        algorithm_val = SVM(
            model=model,
            inner_loss_func=torch.nn.CrossEntropyLoss(),
            n_way=args.n_way_val,
            n_shot=args.n_shot_val,
            n_query=args.n_query_val,
            max_iter=3,
            device='cuda')
    elif args.algorithm == 'Ridge':
        algorithm_train = Ridge(
            model=model,
            inner_loss_func=torch.nn.CrossEntropyLoss(),
            n_way=args.n_way_train,
            n_shot=args.n_shot_train,
            n_query=args.n_query_train,
            device='cuda')
        algorithm_val = Ridge(
            model=model,
            inner_loss_func=torch.nn.CrossEntropyLoss(),
            n_way=args.n_way_val,
            n_shot=args.n_shot_val,
            n_query=args.n_query_val,
            device='cuda')
    else:
        raise ValueError(
            'Unrecognized algorithm {}'.format(args.algorithm))

    trainer = MetaClassical_algorithm_trainer(
        model=model,
        algorithm=algorithm_train,
        optimizer=optimizer,
        writer=writer,
        log_interval=args.log_interval, 
        save_folder=save_folder, 
        grad_clip=args.grad_clip,
        loss_func=torch.nn.CrossEntropyLoss(),
        eps=args.eps
    )


    ####################################################
    #                  META VALIDATOR                  #
    ####################################################

    val_trainer = Generic_adaptation_trainer(
        algorithm=algorithm_val,
        aux_objective=None,
        outer_loss_func=torch.nn.CrossEntropyLoss(),
        outer_optimizer=None, 
        writer=writer,
        log_interval=args.log_interval,
        model_type=args.model_type
    )

    ####################################################
    #                  TRAINER LOOP                    #
    ####################################################


    lambda_epoch = lambda e: 1.0 if e < args.drop_lr_epoch else (0.06 if e < 40 else 0.012 if e < 50 else (0.0024))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda_epoch, last_epoch=-1)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(
    #     optimizer, step_size=2, gamma=0.5, last_epoch=-1)

    for _ in range(args.restart_iter):
        lr_scheduler.step()
    
    for iter_start in range(args.restart_iter, args.n_epochs):

        # training
        for param_group in optimizer.param_groups:
            print('\n\nlearning rate:', param_group['lr'])

        trainer.run(mt_loader, mt_datamgr, is_training=True, epoch=iter_start + 1, n_query=args.n_query_train)

        # validation
        print("Validation")
        results = val_trainer.run(mt_val_loader, mt_val_datamgr)
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(results)
        writer.add_scalar(
            "validation_acc", results['test_loss_after']['accu'], iter_start + 1)
        writer.add_scalar(
            "validation_loss", results['test_loss_after']['loss'], iter_start + 1)
        print("Test")
        results = val_trainer.run(mt_test_loader, mt_test_datamgr)
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(results)
        writer.add_scalar(
            "test_acc", results['test_loss_after']['accu'], iter_start + 1)
        writer.add_scalar(
            "test_loss", results['test_loss_after']['loss'], iter_start + 1)
        # scheduler step
        lr_scheduler.step()
        


if __name__ == '__main__':

    def str2bool(arg):
        return arg.lower() == 'true'

    parser = argparse.ArgumentParser(
        description='Training the feature backbone on all classes from all tasks.')
    
    parser.add_argument('--random-seed', type=int, default=0,
        help='')

    # Algorithm
    parser.add_argument('--algorithm', type=str, help='type of algorithm')

    # Model
    parser.add_argument('--model-type', type=str, default='gatedconv',
        help='type of the model')
    parser.add_argument('--classifier-type', type=str, default='linear',
        help='classifier type [distance based, linear, GDA]')
    parser.add_argument('--loss-names', type=str, nargs='+', default='cross_ent',
        help='names of various loss functions that are part fo overall objective')
    parser.add_argument('--gamma', type=float, default=0.01,
        help='scalar factor multiplied with aux loss')

    # Optimization
    parser.add_argument('--lr', type=float, default=0.001,
        help='learning rate for the global update')
    parser.add_argument('--grad-clip', type=float, default=0.0,
        help='gradient clipping')
    parser.add_argument('--n-epochs', type=int, default=60000,
        help='number of model training epochs')
    parser.add_argument('--weight-decay', type=float, default=0.,
        help='weight decay')
    
    
    # Dataset
    parser.add_argument('--dataset-path', type=str,
        help='which dataset to use')
    parser.add_argument('--img-side-len', type=int, default=84,
        help='width and height of the input images')
    parser.add_argument('--batch-size-train', type=int, default=20,
        help='batch size for training')
    parser.add_argument('--num-classes-train', type=int, default=200,
        help='no of train classes')
    parser.add_argument('--num-classes-val', type=int, default=200,
        help='no of novel (val) classes')
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
    parser.add_argument('--label-offset', type=int, default=0,
        help='offset for label values during fine tuning stage')
    parser.add_argument('--n-query-pool', type=int, default=0,
        help='pool for query samples')
    parser.add_argument('--eps', type=float, default=0.0,
        help='epsilon of label smoothing')

    

    # Miscellaneous
    parser.add_argument('--output-folder', type=str, default='maml',
        help='name of the output folder')
    parser.add_argument('--device-number', type=str, default='0',
        help='gpu device number')
    parser.add_argument('--log-interval', type=int, default=100,
        help='number of batches between tensorboard writes')
    parser.add_argument('--checkpoint', type=str, default='',
        help='path to saved parameters.')
    parser.add_argument('--eval', action='store_true', default=False,
        help='evaluate model')
    parser.add_argument('--train-aug', action='store_true', default=False,
        help='perform data augmentation during training')
    parser.add_argument('--support-aug', action='store_true', default=False,
        help='perform data augmentation on support set')
    parser.add_argument('--n-iterations-val', type=int, default=100,
        help='no. of iterations validation.') 
    parser.add_argument('--restart-iter', type=int, default=0,
        help='iteration at restart') 
    parser.add_argument('--n-tf-updates', type=int, default=1,
        help='no of tf updates before meta update') 
    parser.add_argument('--fix-support', type=int, default=0,
        help='fix support set')
    parser.add_argument('--classifier-metric', type=str, default='',
        help='')
    parser.add_argument('--projection', type=str, default='',
        help='')
    parser.add_argument('--load-optimizer', action='store_true', default=False,
        help='load opt from chkpt')
    parser.add_argument('--drop-lr-epoch', type=int, default=20,
        help='')
    parser.add_argument('--avg-pool', type=str, default='True',
        help='')

    args = parser.parse_args()

    # set random seed. only set for numpy, uncomment the below lines for torch and CuDNN.
    if args.random_seed != 0:
        np.random.seed(args.random_seed)
    # torch.manual_seed(args.random_seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    
    # Create logs and saves folder if they don't exist
    if not os.path.exists('./train_dir_2'):
        os.makedirs('./train_dir_2')

    # main function call
    main(args)
