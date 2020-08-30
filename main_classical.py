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


from algorithm_trainer.models import gated_conv_net_original, resnet, resnet_2, wide_resnet, resnet_12, res_mix_up, conv64
from algorithm_trainer.algorithm_trainer import Classical_algorithm_trainer, Generic_adaptation_trainer
from algorithm_trainer.algorithms.algorithm import SVM, ProtoNet, Finetune, ProtoCosineNet
from algorithm_trainer.utils import optimizer_to_device
from algorithm_trainer.algorithms import modified_sgd
from data_layer.dataset_managers import ClassicalDataManager, MetaDataManager
from analysis.objectives import var_reduction_disc, var_reduction_ortho, rfc_and_pc, ortho_directions, rfc
from algorithm_trainer.utils import *

def main(args):
    is_training = not args.eval
    run_name = 'train' if is_training else 'eval'

    if is_training:
        writer = SummaryWriter('./train_dir/{0}/{1}'.format(
            args.output_folder, run_name))
        with open('./train_dir/{}/config.txt'.format(
            args.output_folder), 'w') as config_txt:
            for k, v in sorted(vars(args).items()):
                config_txt.write('{}: {}\n'.format(k, v))
    else:
        writer = None

    save_folder = './train_dir/{0}'.format(args.output_folder)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)


    ####################################################
    #         DATASET AND DATALOADER CREATION          #
    ####################################################

    # There are 3 tyoes of files: base, val, novel
    # Here we train on base and validate on val
    image_size = args.img_side_len
    train_file = os.path.join(args.dataset_path, 'base.json')
    train_datamgrs = [
        MetaDataManager(image_size, batch_size=1, n_episodes=1000,
            n_way=args.n_way_train, n_shot=args.n_query_train, n_query=0),
        # ClassicalDataManager(image_size, batch_size=args.batch_size_train),
        MetaDataManager(image_size, batch_size=1, n_episodes=100000000,
            n_way=args.n_way_train, n_shot=args.n_shot_train, n_query=0, fix_support=args.fix_support)
    ]
    train_loaders = [
        train_datamgrs[0].get_data_loader(train_file, aug=args.train_aug),
        train_datamgrs[1].get_data_loader(train_file, aug=False)
    ]
    val_file = os.path.join(args.dataset_path, 'val.json')
    meta_val_datamgr = MetaDataManager(
        image_size, batch_size=args.batch_size_val, n_episodes=args.n_iterations_val,
        n_way=args.n_way_val, n_shot=args.n_shot_val, n_query=args.n_query_val)
    meta_val_loader = meta_val_datamgr.get_data_loader(val_file, aug=False)
    
    

    ####################################################
    #             MODEL/BACKBONE CREATION              #
    ####################################################

    if args.model_type == 'conv64':
        model = conv64.Conv64(num_classes=args.num_classes_train, 
            classifier_type=args.classifier_type)
    elif args.model_type == 'wide_resnet':
        model = wide_resnet.wrn28_10(num_classes=args.num_classes_train, 
            classifier_type=args.classifier_type)
    elif args.model_type == 'resnet':
        model = resnet_2.ResNet18(num_classes=args.num_classes_train,
            classifier_type=args.classifier_type)
    elif args.model_type == 'resnet12':
        if args.dataset_path.split('/')[-1] in ['miniImagenet', 'CUB']:
            model = resnet_12.resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=5,
                num_classes=args.num_classes_train, classifier_type=args.classifier_type,
                projection=False, classifier_metric='euclidean', lambd=args.lambd)
        else:
            model = resnet_12.resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=2,
                num_classes=args.num_classes_train, classifier_type=args.classifier_type, 
                projection=False, classifier_metric='euclidean', lambd=args.lambd)
    elif args.model_type == 'res_mix_up':
        model = res_mix_up.resnet18(num_classes=args.num_classes_train,
            classifier_type=args.classifier_type)
    elif args.model_type == 'conv64':
        model = ImpRegConvModel(
            input_channels=3, num_channels=64, img_side_len=image_size, num_classes=args.num_classes_train,
            verbose=True, retain_activation=True, use_group_norm=True, add_bias=False)
    else:
        raise ValueError(
            'Unrecognized model type {}'.format(args.model_type))
    print("Model\n" + "=="*27)    
    print(model)        


    ####################################################
    #                OPTIMIZER CREATION                #
    ####################################################

    # loss_dict = {
    #     'cross_ent': torch.nn.CrossEntropyLoss(),
    #     'var_disc': var_reduction_disc
    # }
    # loss_funcs = []
    # for loss_name, lambda_value in zip(args.loss_names, args.lambdas):
    #     print(f"Adding loss : {loss_name} with lambda value {lambda_value}")
    #     loss_funcs.append((loss_name, loss_dict[loss_name]))
    
    # all_params = []
    # exc_params = []
    # for name, param in model.named_parameters():
    #     if 'fc.Lglinear' not in name:
    #         all_params.append(param)
    #     else:
    #         exc_params.append(param)
    #         print(f"excluding {name}")

    optimizer = modified_sgd.SGD([
        {'params': model.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay, 
            'momentum': args.momentum, 'nesterov': True},
        # {'params': exc_params , 'lr': 0.01, 'weight_decay': 0.},
    ])
    # optimizer = torch.optim.Adam([
    #     {'params': model.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay},
    # ])
    print("Total n_epochs: ", args.n_epochs)


    ####################################################
    #               load from checkpoint               #
    ####################################################


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
    #                CLASSICAL TRAINER                 #
    ####################################################

    trainer = Classical_algorithm_trainer(
        model=model,
        optimizer=optimizer,
        writer=writer,
        log_interval=args.log_interval, 
        save_folder=save_folder, 
        grad_clip=args.grad_clip,
        loss=('cross_ent', torch.nn.CrossEntropyLoss()),
        # loss=('margin loss', torch.nn.MultiMarginLoss(p=2, margin=10)),
        # aux_loss=('rfc', rfc),
        gamma=args.gamma,
        update_gap=1,
        eps=args.eps,
        num_classes=args.num_classes_train
    )

    ####################################################
    #                  META VALIDATOR                  #
    ####################################################


    # swa model
    # cycle = 5
    # print(f"Using cycle {cycle} and initializing swa model with model")
    # swa_model = resnet_12.resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=5,
    #         num_classes=args.num_classes_train, classifier_type=args.classifier_type)
    # # # swa_model.load_state_dict(self._model.state_dict())
    # swa_model.cuda()
    # swa_model = torch.nn.DataParallel(swa_model, device_ids=range(torch.cuda.device_count()))
    


    # algorithm
    if args.algorithm == 'ProtonetCosine':
        algorithm = ProtoCosineNet(
            # model=swa_model,
            model=model,
            inner_loss_func=torch.nn.CrossEntropyLoss(),
            n_way=args.n_way_val,
            n_shot=args.n_shot_val,
            n_query=args.n_query_val,
            device='cuda')

    elif args.algorithm == 'SVM':
        algorithm = SVM(
            model=model,
            inner_loss_func=torch.nn.CrossEntropyLoss(),
            n_way=args.n_way_val,
            n_shot=args.n_shot_val,
            n_query=args.n_query_val,
            device='cuda')

    elif args.algorithm == 'Protonet':
        algorithm = ProtoNet(
            model=model,
            inner_loss_func=torch.nn.CrossEntropyLoss(),
            n_way=args.n_way_val,
            n_shot=args.n_shot_val,
            n_query=args.n_query_val,
            device='cuda')
    elif args.algorithm == 'Finetune':
        algorithm = Finetune(
            model=model,
            inner_loss_func=torch.nn.CrossEntropyLoss(),
            n_updates=500,
            classifier_type=args.classifier_type,
            aux_loss=var_reduction_ortho,
            final_feat_dim=model.module.final_feat_dim,
            n_way=args.n_way_val,
            device='cuda')

    else:
        raise ValueError(
            'Unrecognized algorithm {}'.format(args.algorithm))



    val_trainer = Generic_adaptation_trainer(
        algorithm=algorithm,
        aux_objective=None,
        outer_loss_func=torch.nn.CrossEntropyLoss(),
        # outer_loss_func=torch.nn.MultiMarginLoss(p=2, margin=10),
        outer_optimizer=optimizer, 
        writer=writer,
        log_interval=150,
        model_type=args.model_type,
        n_aux_objective_steps=0,
        label_offset=args.label_offset
    )

    ####################################################
    #                  TRAINER LOOP                    #
    ####################################################




    # lambda_epoch = lambda e: 1.0 if e < 120  else (0.2 if e < 360 else (0.04))
    # lambda_epoch = lambda e: 1.0 if e < 100  else (0.1 if e < 200 else (0.01))
    # lambda_epoch = lambda e: 1.0 if e < 20 else (0.06 if e < 60 else 0.012 if e < 120 else (0.0024))
    # lambda_epoch = lambda e: 1.0 if e < 10 else (0.1 if e < 40 else (0.01 if e < 60 else (0.002)))
    lambda_epoch = lambda e: 1.0 if e < 10 else (0.06 if e < 40 else 0.012 if e < 50 else (0.0024))
    # lambda_epoch = lambda e: 1.0 if e < 80 else (0.1 if e < 200 else (0.01 if e < 280 else (0.002)))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda_epoch, last_epoch=-1)
    

    

    # opt_lambd = torch.optim.SGD([model.module.fc.Lglinear], lr=5e-3)

    # n_s = 3
    # count = 0

    # fix support 
    aux_iter = iter(train_loaders[1])
    aux_batch_x, aux_batch_y = next(aux_iter)
    # print("Fixing randomly chosen support set")

    for iter_start in range(args.restart_iter, args.n_epochs):

        # if iter_start % 50 == 0 and n_s < 3 and model.module.fc.lambd == 0.:
        #     n_s += 1
        #     print(f"Increasing size of support set to {n_s}")
        #     train_datamgrs[1] = MetaDataManager(image_size, batch_size=1, n_episodes=100000000,
        #         n_way=args.n_way_train, n_shot=n_s, n_query=0)
        #     train_loaders[1] = train_datamgrs[1].get_data_loader(train_file, aug=False)

        # if iter_start % 10 == 0:
        #     print("Fixing randomly chosen support set")
        #     aux_batch_x, aux_batch_y = next(aux_iter)

        if args.classifier_type == 'avg-classifier':
            model.module.fc.update_lambd()
            print("scale factor: ", model.module.fc.scale_factor)

        # if iter_start % cycle == 0 and iter_start != 0:
        #     print(f"Computing SWA Model")
        #     count += 1
        #     swa_model = get_swa_model(swa_model, model, 1./count) 
        #     bn_update(train_loaders[0], swa_model)
        #     print("swa model evaluation")
        #     results = val_trainer.run(meta_val_loader, meta_val_datamgr)
        #     pp = pprint.PrettyPrinter(indent=4)
        #     pp.pprint(results)
        #     print("Fixing randomly chosen support set")
        #     aux_batch_x, aux_batch_y = next(aux_iter)
        

        # training
        for param_group in optimizer.param_groups:
            # param_group['lr'] = set_lr(param_group['lr'], iter_start, cycle, args.lr, 0.001)
            print('\n\nlearning rate:', param_group['lr'])
        trainer.run(train_loaders, aux_batch_x, aux_batch_y, is_training=True, epoch=iter_start)
        # if iter_start % 10 == 0:
        #     trainer.run(train_loaders, aux_batch_x, aux_batch_y, is_training=True, epoch=iter_start, grad_analysis=True)

        # validation
        if iter_start % 1 == 0:
            model.module.scale = torch.nn.Parameter(torch.tensor([1.0], device='cuda'))
            results = val_trainer.run(meta_val_loader, meta_val_datamgr)
            pp = pprint.PrettyPrinter(indent=4)
            pp.pprint(results)
    
        # scheduler step
        lr_scheduler.step()



if __name__ == '__main__':

    def str2bool(arg):
        return arg.lower() == 'true'

    parser = argparse.ArgumentParser(
        description='Training the feature backbone on all classes from all tasks.')

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
    parser.add_argument('--momentum', type=float, default=0.9,
        help='SGD momentum')
    parser.add_argument('--load-optimizer', action='store_true', default=False,
        help='load opt from chkpt')
    
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
    parser.add_argument('--train-aug', action='store_true', default=True,
        help='perform data augmentation during training')
    parser.add_argument('--n-iterations-val', type=int, default=100,
        help='no. of iterations validation.') 
    parser.add_argument('--restart-iter', type=int, default=0,
        help='iteration at restart') 
    parser.add_argument('--lowdim', type=int, default=0,
        help='low dim projection') 
    parser.add_argument('--eps', type=float, default=0.0,
        help='epsilon of label smoothing')
    parser.add_argument('--fix-support', action='store_true', default=False,
        help='fix support set')
    parser.add_argument('--lambd', type=float, default=0.0,
        help='lambda of cvx combination')
    
    args = parser.parse_args()

    # Create logs and saves folder if they don't exist
    if not os.path.exists('./train_dir'):
        os.makedirs('./train_dir')

    # main function call
    main(args)