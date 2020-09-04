import os
from tqdm import tqdm
import json
import argparse
import pickle
import torch
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
import pprint
import re

from algorithm_trainer.models import gated_conv_net_original, resnet, resnet_2, resnet_12
from algorithm_trainer.algorithm_trainer import Generic_algorithm_trainer, LR_algorithm_trainer, Classical_algorithm_trainer
from algorithm_trainer.algorithms.algorithm import LR, SVM, ProtoNet, ProtoCosineNet
from algorithm_trainer.utils import optimizer_to_device, add_fc, remove_fc
from data_layer.dataset_managers import MetaDataManager




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
    val_file = os.path.join(args.dataset_path, 'val.json')
    train_datamgr = MetaDataManager(
        image_size, batch_size=args.batch_size_train, n_episodes=args.n_iterations_train,
        n_way=args.n_way_train, n_shot=args.n_shot_train, n_query=args.n_query_train)
    train_loader = train_datamgr.get_data_loader(train_file, aug=args.train_aug)
    val_datamgr = MetaDataManager(
        image_size, batch_size=args.batch_size_val, n_episodes=args.n_iterations_val,
        n_way=args.n_way_val, n_shot=args.n_shot_val, n_query=args.n_query_val)
    val_loader = val_datamgr.get_data_loader(val_file, aug=False)
    


    ####################################################
    #             MODEL/BACKBONE CREATION              #
    ####################################################

    if args.model_type == 'resnet':
        model = resnet_2.ResNet18(no_fc_layer=args.no_fc_layer, add_bias=args.add_bias)
    elif args.model_type == 'resnet12':
        model = resnet_12.resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=5,
            no_fc_layer=args.no_fc_layer, add_bias=args.add_bias)
    elif args.model_type == 'conv64':
        model = ImpRegConvModel(
            num_channels=64, verbose=True, retain_activation=True, 
            use_group_norm=True, add_bias=args.add_bias, no_fc_layer=args.no_fc_layer)
    else:
        raise ValueError(
            'Unrecognized model type {}'.format(args.model_type))
    print("Model\n" + "=="*27)    
    print(model)
    prefc_feature_sz = model.final_feat_dim


    # load from checkpoint
    if args.checkpoint != '':
        print(f"loading from {args.checkpoint}")
        model_dict = model.state_dict()
        chkpt_state_dict = torch.load(args.checkpoint)
        if 'model' in chkpt_state_dict:
            chkpt_state_dict = chkpt_state_dict['model']
        chkpt_state_dict_cpy = chkpt_state_dict.copy()
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
                    
        
    # Multi-gpu support and device setup
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_number
    print('Using GPUs: ', os.environ["CUDA_VISIBLE_DEVICES"])
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.cuda()




    ####################################################
    #                OPTIMIZER CREATION                #
    ####################################################


    loss_func = nn.CrossEntropyLoss()
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD([
            {'params': model.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay, 
                'momentum': 0.9, 'nesterov': True},
        ])
        # optimizer = torch.optim.SGD(
        #     model.parameters(), lr=args.lr, 
        #     momentum=0.9, nesterov=True, weight_decay=args.weight_decay)
    print("Total episodes: ", args.n_iterations_train)
    print("Total tasks: ", args.n_iterations_train * args.batch_size_train)
    
        


    ####################################################
    #         ALGORITHM & ALGORITHM TRAINER            #
    ####################################################


    if args.algorithm == 'ProtonetCosine':
        algorithm = ProtoCosineNet(
            model=model,
            inner_loss_func=torch.nn.CrossEntropyLoss(),
            n_way=args.n_way_train,
            n_shot=args.n_shot_train,
            n_query=args.n_query_train,
            device='cuda')
  
    elif args.algorithm == 'LR':
        algorithm = LR(
            model=model,
            embedding_model=embedding_model,
            inner_loss_func=loss_func,
            l2_lambda=args.l2_inner_loop,
            device=args.device,
            is_classification=True)

    elif args.algorithm == 'SVM':
        algorithm = SVM(
            model=model,
            inner_loss_func=loss_func,
            n_way=args.n_way_train,
            n_shot=args.n_shot_train,
            n_query=args.n_query_train,
            device=args.device)

    elif args.algorithm == 'Protonet':
        algorithm = ProtoNet(
            model=model,
            inner_loss_func=loss_func,
            n_way=args.n_way_train,
            n_shot=args.n_shot_train,
            n_query=args.n_query_train,
            device=args.device)



    if args.algorithm in ['LR']:
        trainer = LR_algorithm_trainer(
            algorithm=algorithm,
            outer_loss_func=loss_func,
            outer_optimizer=optimizer, 
            writer=writer,
            log_interval=args.log_interval, save_interval=args.save_interval,
            model_type=args.model_type, save_folder=save_folder, 
            outer_loop_grad_norm=args.grad_clip,
            hessian_inverse=args.hessian_inverse)
        
    elif args.algorithm in ['SVM', 'Protonet', 'ProtonetCosine']:
        trainer = Generic_algorithm_trainer(
            algorithm=algorithm,
            outer_loss_func=loss_func,
            outer_optimizer=optimizer, 
            writer=writer,
            log_interval=args.log_interval, save_interval=args.save_interval,
            save_folder=save_folder, outer_loop_grad_norm=args.grad_clip,
            model_type=args.model_type,
            optimizer_update_interval=args.optimizer_update_interval)

    
    
    if is_training:
        # create train iterators
        epoch_sz = args.n_iterations_train // 1000
        # lambda_epoch = lambda e: 1.0 if e < (epoch_sz // 2) * args.optimizer_update_interval else 0.1
        # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_epoch, last_epoch=-1)
        
        lambda_epoch = lambda e: 1.0 if e < 20 else (0.06 if e < 40 else 0.012 if e < 50 else (0.0024))
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda_epoch, last_epoch=-1)
    
        for iter_start in range(1, args.n_iterations_train, args.val_interval):
        
            # validation
            if hasattr(trainer._algorithm, '_n_way'):
                trainer._algorithm._n_way = args.n_way_val
            if hasattr(trainer._algorithm, '_n_shot'):
                trainer._algorithm._n_shot = args.n_shot_val
            if hasattr(trainer._algorithm, '_n_query'):
                trainer._algorithm._n_query = args.n_query_val
            
            tqdm.write("=="*27+"\nStarting validation")
            val_result = trainer.run(val_loader, val_datamgr, is_training=False, 
            meta_val=True, start=iter_start+args.val_interval-1)
            tqdm.write(str(val_result))
            tqdm.write("Finished validation\n" + "=="*27)

        
            # train
            if hasattr(trainer._algorithm, '_n_way'):
                trainer._algorithm._n_way = args.n_way_train
            if hasattr(trainer._algorithm, '_n_shot'):
                trainer._algorithm._n_shot = args.n_shot_train
            if hasattr(trainer._algorithm, '_n_query'):
                trainer._algorithm._n_query = args.n_query_train

            for param_group in optimizer.param_groups:
                print('optimizer:', args.optimizer, 'lr:', param_group['lr'])
            try:
                train_result = trainer.run(train_loader, train_datamgr, is_training=True, 
                    start=iter_start, stop=iter_start+args.val_interval)
            except StopIteration:
                print("Finished training iterations.")
                print(train_result)
                print("Starting final validation.")
            
            lr_scheduler.step()
            
            
            
    else:
        if hasattr(trainer._algorithm, '_n_way'):
            trainer._algorithm._n_way = args.n_way_val
        if hasattr(trainer._algorithm, '_n_shot'):
            trainer._algorithm._n_shot = args.n_shot_val
        if hasattr(trainer._algorithm, '_n_query'):
            trainer._algorithm._n_query = args.n_query_val
        
        
        # fine tune features on support set of tasks
        if args.fine_tune:
            # add fc layer to model backbone based on number of val classes
            model_with_fc = add_fc(model, prefc_feature_sz, args.num_classes)
            # instantiate classical trainer
            fine_tuner = Classical_algorithm_trainer(
                model=model_with_fc,
                loss_func=loss_func,
                optimizer=optimizer, writer=writer,
                log_interval=args.log_interval, save_folder=save_folder, 
                grad_clip=args.grad_clip
            )
            # fine tune on support samples of meta-validation set, make sure to return all val tasks as is
            fine_tuned_model_with_fc, val_batches = fine_tuner.fine_tune(val_loader, val_datamgr, 
                label_offset=args.label_offset, n_fine_tune_epochs=args.n_fine_tune_epochs)
            # thorow away fc
            fine_tuned_model_without_fc = remove_fc(fine_tuned_model_with_fc)
            # set algorithm's model to be the fine tuned model
            trainer._algorithm._model = fine_tuned_model_without_fc
            # evaluate on query set of val tasks
            results = trainer.run(val_loader, val_datamgr, 
                is_training=False, start=0, fixed_batches=val_batches)   

        else:  
            results = trainer.run(val_loader, val_datamgr, is_training=False, start=0)
        
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(results)
        

if __name__ == '__main__':

    def str2bool(arg):
        return arg.lower() == 'true'

    parser = argparse.ArgumentParser(
        description='Meta-Learning Inner Solvers')

    # Algorithm
    parser.add_argument('--algorithm', type=str, help='type of algorithm')

    # Model
    parser.add_argument('--model-type', type=str, default='gatedconv',
        help='type of the model')
    parser.add_argument('--retain-activation', type=str2bool, default=False,
        help='if True, use activation function in the last layer;\
             otherwise dont use activation in the last layer')
    parser.add_argument('--add-bias', type=str2bool, default=False,
        help='add bias term inner loop')
    parser.add_argument('--no-fc-layer', type=str2bool, default=True,
        help='will not add fc layer to model')
    parser.add_argument('--use-group-norm', type=str2bool, default=False,
        help='use group norm instead of batch norm')
    parser.add_argument('--num-classes', type=int, default=200,
        help='no of classes -- used during fine tuning')
    parser.add_argument('--label-offset', type=int, default=0,
        help='offset for label values during fine tuning stage')
    
    
    # Optimization
    parser.add_argument('--lr', type=float, default=0.001,
        help='learning rate for the global update')
    parser.add_argument('--grad-clip', type=float, default=0.0,
        help='gradient clipping')
    parser.add_argument('--optimizer-update-interval', type=int, default=1,
        help='number of mini batches after which the optimizer is updated')
    parser.add_argument('--optimizer', type=str, default='adam',
        help='optimizer')
    parser.add_argument('--hessian-inverse', type=str2bool, default=False,
        help='for implicit last layer optimization, whether to use \
            hessian to solve linear equation or to use woodbury identity\
            on the hessian inverse')
    parser.add_argument('--n-fine-tune-epochs', type=int, default=60000,
        help='number of model fine tune epochs')
    parser.add_argument('--weight-decay', type=float, default=0.,
        help='weight decay')
    
    
    
    
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
    parser.add_argument('--device', type=str, default='cuda',
        help='set the device (cpu or cuda)')
    parser.add_argument('--device-number', type=str, default='0',
        help='gpu device number')
    parser.add_argument('--log-interval', type=int, default=100,
        help='number of batches between tensorboard writes')
    parser.add_argument('--save-interval', type=int, default=1000,
        help='number of batches between model saves')
    parser.add_argument('--eval', action='store_true', default=False,
        help='evaluate model')
    parser.add_argument('--checkpoint', type=str, default='',
        help='path to saved parameters.')
    parser.add_argument('--val-interval', type=int, default=2000,
        help='no. of iterations after which to perform meta-validation.')
    parser.add_argument('--verbose', type=str2bool, default=False,
        help='debugging purposes')
    parser.add_argument('--n-iterations-train', type=int, default=60000,
        help='no. of iterations train.') 
    parser.add_argument('--n-iterations-val', type=int, default=100,
        help='no. of iterations validation.') 
    parser.add_argument('--train-aug', action='store_true', default=True,
        help='perform data augmentation during training')
    parser.add_argument('--fine-tune', action='store_true', default=False,
        help='fine tune features on the support set at eval time')
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_number
    print('GPU number', os.environ["CUDA_VISIBLE_DEVICES"])

    # Create logs and saves folder if they don't exist
    if not os.path.exists('./train_dir'):
        os.makedirs('./train_dir')
    
    # print args
    if args.verbose:
        print('='*10 + ' ARGS ' + '='*10)
        for k, v in sorted(vars(args).items()):
            print('{}: {}'.format(k, v))
        print('='*26)

    main(args)
