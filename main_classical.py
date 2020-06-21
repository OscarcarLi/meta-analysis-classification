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
from algorithm_trainer.algorithm_trainer import Classical_algorithm_trainer
from algorithm_trainer.utils import optimizer_to_device
from data_layer.dataset_managers import ClassicalDataManager
from data_layer.datasets import ClassicalDataset



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
    val_file = os.path.join(args.dataset_path, 'base.json')
    train_datamgr = ClassicalDataManager(image_size, batch_size=args.train_batch_size)
    train_loader = train_datamgr.get_data_loader(train_file, aug=args.train_aug)
    val_datamgr = ClassicalDataManager(image_size, batch_size=args.val_batch_size)
    val_loader = val_datamgr.get_data_loader(val_file, aug=False)
    

    ####################################################
    #             MODEL/BACKBONE CREATION              #
    ####################################################

    if args.model_type == 'resnet':
        model = resnet.ResNet18(num_classes=args.num_classes, 
            distance_classifier=args.distance_classifier)
    elif args.model_type == 'conv64':
        model = ImpRegConvModel(
            input_channels=3, num_channels=64, img_side_len=image_size,
            verbose=True, retain_activation=True, use_group_norm=True, add_bias=False)
    else:
        raise ValueError(
            'Unrecognized model type {}'.format(args.model_type))
    print("Model\n" + "=="*27)    
    print(model)


    # load from checkpoint
    if args.checkpoint != '':
        print(f"loading from {args.checkpoint}")
        state_dict = torch.load(args.checkpoint)
        model.load_state_dict(state_dict['model'])
        
    # Multi-gpu support and device setup
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_number
    print('Using GPUs: ', os.environ["CUDA_VISIBLE_DEVICES"])
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.cuda()
        


    ####################################################
    #                OPTIMIZER CREATION                #
    ####################################################


    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
    ])
    print("Total n_epochs: ", args.n_epochs)

    ####################################################
    #                  TRAINER LOOP                    #
    ####################################################


    trainer = Classical_algorithm_trainer(
        model=model,
        loss_func=loss_func,
        optimizer=optimizer, writer=writer,
        log_interval=args.log_interval, save_folder=save_folder, 
        grad_norm=args.model_grad_clip
    )
    
    lambda_epoch = lambda e: 1.0 if e < args.n_epochs // 2  else 0.1
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda_epoch, last_epoch=-1)
    
    for iter_start in range(args.n_epochs):
        for param_group in optimizer.param_groups:
            print('\n\nlearning rate:', param_group['lr'])
        train_result = trainer.run(train_loader, is_training=True, epoch=iter_start+1)
        
        # validation
        tqdm.write("\nStarting validation")
        val_result = trainer.run(val_loader, is_training=False)
        tqdm.write("Finished validation\n" + "=="*27)
        
        # scheduler step
        lr_scheduler.step()



if __name__ == '__main__':

    def str2bool(arg):
        return arg.lower() == 'true'

    parser = argparse.ArgumentParser(
        description='Model-Agnostic Meta-Learning (MAML)')

    # Model
    parser.add_argument('--model-type', type=str, default='gatedconv',
        help='type of the model')
    parser.add_argument('--distance-classifier', action='store_true', default=False,
        help='use a distance classifer (cosine based)')

    # Optimization
    parser.add_argument('--lr', type=float, default=0.001,
        help='learning rate for the global update')
    parser.add_argument('--model-grad-clip', type=float, default=0.0,
        help='gradient clipping')
    parser.add_argument('--n-epochs', type=int, default=60000,
        help='number of model training epochs')
    
    # Dataset
    parser.add_argument('--dataset-path', type=str,
        help='which dataset to use')
    parser.add_argument('--train-batch-size', type=int, default=20,
        help='batch size for training')
    parser.add_argument('--val-batch-size', type=int, default=4,
        help='batch size for validation')
    parser.add_argument('--img-side-len', type=int, default=84,
        help='width and height of the input images')
    parser.add_argument('--num-classes', type=int, default=200,
        help='no of classes')
    

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
    
    args = parser.parse_args()

    # Create logs and saves folder if they don't exist
    if not os.path.exists('./train_dir'):
        os.makedirs('./train_dir')

    # main function call
    main(args)