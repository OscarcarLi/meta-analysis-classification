import os
import json
import argparse
import pickle

import torch
import numpy as np
from tensorboardX import SummaryWriter

from maml.datasets.omniglot import OmniglotMetaDataset
from maml.datasets.miniimagenet import MiniimagenetMetaDataset
from maml.datasets.cifar100 import Cifar100MetaDataset
from maml.datasets.bird import BirdMetaDataset
from maml.datasets.aircraft import AircraftMetaDataset
from maml.datasets.multimodal_few_shot import MultimodalFewShotDataset
from maml.models.conv_net import ConvModel
from maml.models.gated_conv_net import GatedConvModel, RegConvModel
from maml.models.simple_embedding_model import SimpleEmbeddingModel
from maml.models.lstm_embedding_model import LSTMEmbeddingModel
from maml.models.gru_embedding_model import GRUEmbeddingModel
from maml.models.conv_embedding_model import ConvEmbeddingModel, RegConvEmbeddingModel
from maml.algorithm import MAML_inner_algorithm, MMAML_inner_algorithm, ModMAML_inner_algorithm, RegMAML_inner_algorithm, ImpRMAML_inner_algorithm
from maml.algorithm_trainer import Gradient_based_algorithm_trainer, Implicit_Gradient_based_algorithm_trainer
from maml.utils import optimizer_to_device, get_git_revision_hash
from maml.models import gated_conv_net_original, gated_conv_net
from maml.models.gated_conv_net import ImpRegConvModel
import pprint


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

    config_name = '{0}_config.json'.format(run_name)
    with open(os.path.join(save_folder, config_name), 'w') as f:
        config = {k: v for (k, v) in vars(args).items() if k != 'device'}
        config.update(device=args.device.type)
        try:
            config.update({'git_hash': get_git_revision_hash()})
        except:
            pass
        json.dump(config, f, indent=2)

    _num_tasks = 1
    # define splits num_batches and num_val samples for the different splits
    # since we need to create different datasets (one for each split)
    dataset_splits = ('train', 'val', 'test')
    num_batches = {
        'train': args.num_batches_meta_train, 
        'val': args.num_batches_meta_val, 
        'test': args.num_batches_meta_test
    }
    num_val_samples_per_class = {
        'train': args.num_val_samples_per_class_meta_train, 
        'val': args.num_val_samples_per_class_meta_val, 
        'test': args.num_val_samples_per_class_meta_test
    }
    dataset = {} # dictionary of datasets, indexed by split

    if args.dataset == 'omniglot':
        for split in dataset_splits:
            dataset[split] = OmniglotMetaDataset(
                root='data',
                img_side_len=28, # args.img_side_len,
                num_classes_per_batch=args.num_classes_per_batch,
                num_samples_per_class=args.num_train_samples_per_class,
                num_total_batches=num_batches[split],
                num_val_samples=num_val_samples_per_class[split],
                meta_batch_size=args.meta_batch_size,
                split=split,
                num_train_classes=args.num_train_classes,
                num_workers=args.num_workers,
                device=args.device)
        loss_func = torch.nn.CrossEntropyLoss()
        collect_accuracies = True
    elif args.dataset == 'cifar':
        for split in dataset_splits:
            dataset[split] = Cifar100MetaDataset(
                root='data',
                img_side_len=32,
                num_classes_per_batch=args.num_classes_per_batch,
                num_samples_per_class=args.num_train_samples_per_class,
                num_total_batches=num_batches[split],
                num_val_samples=num_val_samples_per_class[split],
                meta_batch_size=args.meta_batch_size,
                split=split,
                num_workers=args.num_workers,
                device=args.device)
        loss_func = torch.nn.CrossEntropyLoss()
        collect_accuracies = True
    elif args.dataset == 'multimodal_few_shot':
        for split in dataset_splits:
            # multiple different classes of classification tasks
            dataset_list = []
            # when we mix the datasets together we use the square image with side length common_img_side_len
            if 'omniglot' in args.multimodal_few_shot:
                dataset_list.append(OmniglotMetaDataset(
                    root='data',
                    img_side_len=args.common_img_side_len,
                    img_channel=args.common_img_channel,
                    num_classes_per_batch=args.num_classes_per_batch,
                    num_samples_per_class=args.num_train_samples_per_class,
                    num_total_batches=num_batches[split],
                    num_val_samples=num_val_samples_per_class[split],
                    meta_batch_size=args.meta_batch_size,
                    split=split,
                    num_train_classes=args.num_train_classes,
                    num_workers=args.num_workers,
                    device=args.device)
                )
            if 'miniimagenet' in args.multimodal_few_shot:
                dataset_list.append( MiniimagenetMetaDataset(
                    root='data',
                    img_side_len=args.common_img_side_len,
                    img_channel=args.common_img_channel,
                    num_classes_per_batch=args.num_classes_per_batch,
                    num_samples_per_class=args.num_train_samples_per_class,
                    num_total_batches=num_batches[split],
                    num_val_samples=num_val_samples_per_class[split],
                    meta_batch_size=args.meta_batch_size,
                    split=split,
                    num_workers=args.num_workers,
                    device=args.device)
                )           
            if 'cifar' in args.multimodal_few_shot:
                dataset_list.append(Cifar100MetaDataset(
                    root='data',
                    img_side_len=args.common_img_side_len,
                    img_channel=args.common_img_channel,
                    num_classes_per_batch=args.num_classes_per_batch,
                    num_samples_per_class=args.num_train_samples_per_class,
                    num_total_batches=num_batches[split],
                    num_val_samples=num_val_samples_per_class[split],
                    meta_batch_size=args.meta_batch_size,
                    split=split,
                    num_workers=args.num_workers,
                    device=args.device)
                )
            # if 'doublemnist' in args.multimodal_few_shot:
            #     dataset_list.append( DoubleMNISTMetaDataset(
            #         root='data',
            #         img_side_len=args.common_img_side_len,
            #         img_channel=args.common_img_channel,
            #         num_classes_per_batch=args.num_classes_per_batch,
            #         num_samples_per_class=args.num_samples_per_class,
            #         num_total_batches=args.num_batches,
            #         num_val_samples=args.num_val_samples,
            #         meta_batch_size=args.meta_batch_size,
            #         train=is_training,
            #         num_train_classes=args.num_train_classes,
            #         num_workers=args.num_workers,
            #         device=args.device)
            #     )
            # if 'triplemnist' in args.multimodal_few_shot:
            #     dataset_list.append( TripleMNISTMetaDataset(
            #         root='data',
            #         img_side_len=args.common_img_side_len,
            #         img_channel=args.common_img_channel,
            #         num_classes_per_batch=args.num_classes_per_batch,
            #         num_samples_per_class=args.num_samples_per_class,
            #         num_total_batches=args.num_batches,
            #         num_val_samples=args.num_val_samples,
            #         meta_batch_size=args.meta_batch_size,
            #         train=is_training,
            #         num_train_classes=args.num_train_classes,
            #         num_workers=args.num_workers,
            #         device=args.device)
            #     )
            if 'bird' in args.multimodal_few_shot:
                dataset_list.append(BirdMetaDataset(
                    root='data',
                    img_side_len=args.common_img_side_len,
                    img_channel=args.common_img_channel,
                    num_classes_per_batch=args.num_classes_per_batch,
                    num_samples_per_class=args.num_train_samples_per_class,
                    num_total_batches=num_batches[split],
                    num_val_samples=num_val_samples_per_class[split],
                    meta_batch_size=args.meta_batch_size,
                    split=split,
                    num_workers=args.num_workers,
                    device=args.device)
                )           
            if 'aircraft' in args.multimodal_few_shot:
                dataset_list.append(AircraftMetaDataset(
                    root='data',
                    img_side_len=args.common_img_side_len,
                    img_channel=args.common_img_channel,
                    num_classes_per_batch=args.num_classes_per_batch,
                    num_samples_per_class=args.num_train_samples_per_class,
                    num_total_batches=num_batches[split],
                    num_val_samples=num_val_samples_per_class[split],
                    meta_batch_size=args.meta_batch_size,
                    split=split,
                    num_workers=args.num_workers,
                    device=args.device)
                )           
            assert len(dataset_list) > 0
            print('Multimodal Few Shot Datasets: {}'.format(
                ' '.join([dataset.name for dataset in dataset_list])))
            dataset[split] = MultimodalFewShotDataset(
                dataset_list, 
                num_total_batches=num_batches[split],
                mix_meta_batch=args.mix_meta_batch,
                mix_mini_batch=args.mix_mini_batch,
                txt_file=args.sample_embedding_file+'.txt' if args.num_sample_embedding > 0 else None,
                split=split,
            )
        loss_func = torch.nn.CrossEntropyLoss()
        collect_accuracies = True
    # elif args.dataset == 'doublemnist':
    #     dataset = DoubleMNISTMetaDataset(
    #         root='data',
    #         img_side_len=64,
    #         num_classes_per_batch=args.num_classes_per_batch,
    #         num_samples_per_class=args.num_samples_per_class,
    #         num_total_batches=args.num_batches,
    #         num_val_samples=args.num_val_samples,
    #         meta_batch_size=args.meta_batch_size,
    #         train=is_training,
    #         num_train_classes=args.num_train_classes,
    #         num_workers=args.num_workers,
    #         device=args.device)
    #     loss_func = torch.nn.CrossEntropyLoss()
    #     collect_accuracies = True
    # elif args.dataset == 'triplemnist':
    #     dataset = TripleMNISTMetaDataset(
    #         root='data',
    #         img_side_len=84,
    #         num_classes_per_batch=args.num_classes_per_batch,
    #         num_samples_per_class=args.num_samples_per_class,
    #         num_total_batches=args.num_batches,
    #         num_val_samples=args.num_val_samples,
    #         meta_batch_size=args.meta_batch_size,
    #         train=is_training,
    #         num_train_classes=args.num_train_classes,
    #         num_workers=args.num_workers,
    #         device=args.device)
    #     loss_func = torch.nn.CrossEntropyLoss()
    #     collect_accuracies = True
    elif args.dataset == 'miniimagenet':
        for split in dataset_splits:
            dataset[split] = MiniimagenetMetaDataset(
                root='data',
                img_side_len=84,
                num_classes_per_batch=args.num_classes_per_batch,
                num_samples_per_class=args.num_train_samples_per_class,
                num_total_batches=num_batches[split],
                num_val_samples=num_val_samples_per_class[split],
                meta_batch_size=args.meta_batch_size,
                split=split,
                num_workers=args.num_workers,
                device=args.device)
        loss_func = torch.nn.CrossEntropyLoss()
        collect_accuracies = True
    # elif args.dataset == 'sinusoid':
    #     dataset = SinusoidMetaDataset(
    #         num_total_batches=args.num_batches,
    #         num_samples_per_function=args.num_samples_per_class,
    #         num_val_samples=args.num_val_samples,
    #         meta_batch_size=args.meta_batch_size,
    #         amp_range=args.amp_range,
    #         phase_range=args.phase_range,
    #         input_range=args.input_range,
    #         oracle=args.oracle,
    #         train=is_training,
    #         device=args.device)
    #     loss_func = torch.nn.MSELoss()
    #     collect_accuracies = False
    # elif args.dataset == 'linear':
    #     dataset = LinearMetaDataset(
    #         num_total_batches=args.num_batches,
    #         num_samples_per_function=args.num_samples_per_class,
    #         num_val_samples=args.num_val_samples,
    #         meta_batch_size=args.meta_batch_size,
    #         slope_range=args.slope_range,
    #         intersect_range=args.intersect_range,
    #         input_range=args.input_range,
    #         oracle=args.oracle,
    #         train=is_training,
    #         device=args.device)
    #     loss_func = torch.nn.MSELoss()
    #     collect_accuracies = False
    # elif args.dataset == 'mixed':
    #     dataset = MixedFunctionsMetaDataset(
    #         num_total_batches=args.num_batches,
    #         num_samples_per_function=args.num_samples_per_class,
    #         num_val_samples=args.num_val_samples,
    #         meta_batch_size=args.meta_batch_size,
    #         amp_range=args.amp_range,
    #         phase_range=args.phase_range,
    #         slope_range=args.slope_range,
    #         intersect_range=args.intersect_range,
    #         input_range=args.input_range,
    #         noise_std=args.noise_std,
    #         oracle=args.oracle,
    #         task_oracle=args.task_oracle,
    #         train=is_training,
    #         device=args.device)
    #     loss_func = torch.nn.MSELoss()
    #     collect_accuracies = False
    #     _num_tasks=2
    # elif args.dataset == 'many':
    #     dataset = ManyFunctionsMetaDataset(
    #         num_total_batches=args.num_batches,
    #         num_samples_per_function=args.num_samples_per_class,
    #         num_val_samples=args.num_val_samples,
    #         meta_batch_size=args.meta_batch_size,
    #         amp_range=args.amp_range,
    #         phase_range=args.phase_range,
    #         slope_range=args.slope_range,
    #         intersect_range=args.intersect_range,
    #         input_range=args.input_range,
    #         noise_std=args.noise_std,
    #         oracle=args.oracle,
    #         task_oracle=args.task_oracle,
    #         train=is_training,
    #         device=args.device)
    #     loss_func = torch.nn.MSELoss()
    #     collect_accuracies = False
    #     _num_tasks=3
    # elif args.dataset == 'multisinusoids':
    #     dataset = MultiSinusoidsMetaDataset(
    #         num_total_batches=args.num_batches,
    #         num_samples_per_function=args.num_samples_per_class,
    #         num_val_samples=args.num_val_samples,
    #         meta_batch_size=args.meta_batch_size,
    #         amp_range=args.amp_range,
    #         phase_range=args.phase_range,
    #         slope_range=args.slope_range,
    #         intersect_range=args.intersect_range,
    #         input_range=args.input_range,
    #         noise_std=args.noise_std,
    #         oracle=args.oracle,
    #         task_oracle=args.task_oracle,
    #         train=is_training,
    #         device=args.device)
    #     loss_func = torch.nn.MSELoss()
    #     collect_accuracies = False
    else:
        raise ValueError('Unrecognized dataset {}'.format(args.dataset))

    embedding_model = None

    # if args.model_type == 'fc':
    #     model = FullyConnectedModel(
    #         input_size=np.prod(dataset.input_size),
    #         output_size=dataset.output_size,
    #         hidden_sizes=args.hidden_sizes,
    #         disable_norm=args.disable_norm,
    #         bias_transformation_size=args.bias_transformation_size)
    # elif args.model_type == 'multi':
    #     model = MultiFullyConnectedModel(
    #         input_size=np.prod(dataset.input_size),
    #         output_size=dataset.output_size,
    #         hidden_sizes=args.hidden_sizes,
    #         disable_norm=args.disable_norm,
    #         num_tasks=_num_tasks,
    #         bias_transformation_size=args.bias_transformation_size)
    # elif args.model_type == 'conv':
    if args.model_type == 'conv':
        model = ConvModel(
            input_channels=dataset['train'].input_size[0],
            output_size=dataset['train'].output_size,
            num_channels=args.num_channels,
            img_side_len=dataset['train'].input_size[1],
            use_max_pool=args.use_max_pool,
            verbose=args.verbose)
    elif args.model_type == 'gatedconv':
        model = GatedConvModel(
            input_channels=dataset['train'].input_size[0],
            output_size=dataset['train'].output_size,
            use_max_pool=args.use_max_pool,
            num_channels=args.num_channels,
            img_side_len=dataset['train'].input_size[1],
            condition_type=args.condition_type,
            condition_order=args.condition_order,
            verbose=args.verbose)
    elif args.model_type == 'gated':
        model = GatedNet(
            input_size=np.prod(dataset['train'].input_size),
            output_size=dataset['train'].output_size,
            hidden_sizes=args.hidden_sizes,
            condition_type=args.condition_type,
            condition_order=args.condition_order)
    elif args.model_type == 'regconv':
        if args.original_conv:
            print("Using original MAML implementation")
            model = gated_conv_net_original.RegConvModel(
                input_channels=dataset['train'].input_size[0],
                output_size=dataset['train'].output_size,
                num_channels=args.num_channels,
                modulation_mat_rank=args.modulation_mat_rank,
                img_side_len=dataset['train'].input_size[1],
                use_max_pool=args.use_max_pool,
                verbose=args.verbose)
        else:
            model = gated_conv_net.RegConvModel(
                input_channels=dataset['train'].input_size[0],
                output_size=dataset['train'].output_size,
                num_channels=args.num_channels,
                modulation_mat_rank=args.modulation_mat_rank,
                img_side_len=dataset['train'].input_size[1],
                use_max_pool=args.use_max_pool,
                verbose=args.verbose)
    elif args.model_type == 'impregconv':
        model = ImpRegConvModel(
                input_channels=dataset['train'].input_size[0],
                output_size=dataset['train'].output_size,
                num_channels=args.num_channels,
                modulation_mat_rank=args.modulation_mat_rank,
                img_side_len=dataset['train'].input_size[1],
                use_max_pool=args.use_max_pool,
                verbose=args.verbose)
    else:
        raise ValueError('Unrecognized model type {}'.format(args.model_type))
    model_parameters = list(model.parameters())

    if args.embedding_type == '':
        embedding_model = None
    elif args.embedding_type == 'simple':
        embedding_model = SimpleEmbeddingModel(
            num_embeddings=dataset['train'].num_tasks,
            embedding_dims=args.embedding_dims)
        embedding_parameters = list(embedding_model.parameters())
    elif args.embedding_type == 'GRU':
        embedding_model = GRUEmbeddingModel(
             input_size=np.prod(dataset['train'].input_size),
             output_size=dataset['train'].output_size,
             modulation_dims=args.modulation_dims,
             hidden_size=args.embedding_hidden_size,
             num_layers=args.embedding_num_layers)
        embedding_parameters = list(embedding_model.parameters())
    elif args.embedding_type == 'LSTM':
        embedding_model = LSTMEmbeddingModel(
             input_size=np.prod(dataset['train'].input_size),
             output_size=dataset['train'].output_size,
             modulation_dims=args.modulation_dims,
             hidden_size=args.embedding_hidden_size,
             num_layers=args.embedding_num_layers)
        embedding_parameters = list(embedding_model.parameters())
    elif args.embedding_type == 'ConvGRU':
        embedding_model = ConvEmbeddingModel(
             img_size=dataset['train'].input_size,
             modulation_dims=args.modulation_dims,
             use_label=args.use_label,
             num_classes=dataset['train'].output_size,
             hidden_size=args.embedding_hidden_size,
             num_layers=args.embedding_num_layers,
             convolutional=args.conv_embedding,
             num_conv=args.num_conv_embedding_layer,
             num_channels=args.num_channels,
             rnn_aggregation=(not args.no_rnn_aggregation),
             linear_before_rnn=args.linear_before_rnn,
             embedding_pooling=args.embedding_pooling,
             batch_norm=args.conv_embedding_batch_norm,
             avgpool_after_conv=args.conv_embedding_avgpool_after_conv,
             num_sample_embedding=args.num_sample_embedding,
             sample_embedding_file=args.sample_embedding_file+'.'+args.sample_embedding_file_type,
             verbose=args.verbose)
        embedding_parameters = list(embedding_model.parameters())
    elif args.embedding_type == 'RegConvGRU':
        if args.original_conv:
            modulation_mat_size = (args.modulation_mat_rank, args.num_channels*5*5)
            print("modulation_mat_size", modulation_mat_size)
        else:
            modulation_mat_size = (args.modulation_mat_rank, args.num_channels*8)
        embedding_model = RegConvEmbeddingModel(
             input_size=np.prod(dataset['train'].input_size),
             output_size=dataset['train'].output_size,
             modulation_mat_size=modulation_mat_size,
             hidden_size=args.embedding_hidden_size,
             num_layers=args.embedding_num_layers,
             convolutional=args.conv_embedding,
             num_conv=args.num_conv_embedding_layer,
             num_channels=args.num_channels,
             rnn_aggregation=(not args.no_rnn_aggregation),
             linear_before_rnn=args.linear_before_rnn,
             embedding_pooling=args.embedding_pooling,
             batch_norm=args.conv_embedding_batch_norm,
             avgpool_after_conv=args.conv_embedding_avgpool_after_conv,
             num_sample_embedding=args.num_sample_embedding,
             sample_embedding_file=args.sample_embedding_file+'.'+args.sample_embedding_file_type,
             img_size=dataset['train'].input_size,
             verbose=args.verbose,
             original_conv=args.original_conv,
             modulation_mat_spec_norm = args.modulation_mat_spec_norm)
        embedding_parameters = list(embedding_model.parameters())
    else:
        raise ValueError('Unrecognized embedding type {}'.format(
            args.embedding_type))
    print("Model:")
    print(model)
    print("Embedding Model:")
    print(embedding_model)
    optimizers = None
    if embedding_model is None:
        optimizers = torch.optim.Adam(model.parameters(), lr=args.slow_lr)
    else:
        optimizer_specs = \
            [{'params': model.parameters(), 'lr': args.slow_lr},
             {'params': embedding_model.parameters(), 'lr': args.slow_lr}]
        optimizers = torch.optim.Adam(optimizer_specs)

    if args.checkpoint != '':
        print(f"loading from {args.checkpoint}")
        state_dict = torch.load(args.checkpoint)
        model.load_state_dict(state_dict['model'])
        model.to(args.device)
        if embedding_model is not None:
            embedding_model.load_state_dict(
                state_dict['embedding_model'])

    if args.algorithm == 'maml' or args.algorithm == 'op_maml' or args.algorithm == 'cons_op_maml_all': 
        algorithm = MAML_inner_algorithm(
            model=model,
            inner_loss_func=loss_func,
            fast_lr=args.fast_lr,
            first_order=args.first_order,
            num_updates=args.num_updates,
            inner_loop_grad_clip=args.inner_loop_grad_clip,
            inner_loop_soft_clip_slope=args.inner_loop_soft_clip_slope,
            device=args.device,
            is_classification=True)

    # elif args.algorithm == 'cons_op_maml':
    #     algorithm = ModMAML_inner_algorithm(
    #         model=model,
    #         layer_modulations=layer_modulations,
    #         inner_loss_func=loss_func,
    #         fast_lr=args.fast_lr,
    #         first_order=args.first_order,
    #         num_updates=args.num_updates,
    #         inner_loop_grad_clip=args.inner_loop_grad_clip,
    #         inner_loop_soft_clip_slope=args.inner_loop_soft_clip_slope,
    #         device=args.device)

    elif args.algorithm == 'mmaml' or args.algorithm == 'attention_mmaml':
        algorithm = MMAML_inner_algorithm(
            model=model,
            embedding_model=embedding_model,
            inner_loss_func=loss_func,
            fast_lr=args.fast_lr,
            first_order=args.first_order,
            num_updates=args.num_updates,
            inner_loop_grad_clip=args.inner_loop_grad_clip,
            inner_loop_soft_clip_slope=args.inner_loop_soft_clip_slope,
            device=args.device,
            is_classification=True)
    elif args.algorithm == 'reg_maml':
        algorithm = RegMAML_inner_algorithm(
            model=model,
            embedding_model=embedding_model,
            inner_loss_func=loss_func,
            fast_lr=args.fast_lr,
            first_order=args.first_order,
            num_updates=args.num_updates,
            inner_loop_grad_clip=args.inner_loop_grad_clip,
            inner_loop_soft_clip_slope=args.inner_loop_soft_clip_slope,
            device=args.device,
            is_classification=True,
            is_momentum=args.momentum,
            gamma_momentum=args.gamma_momentum,
            l2_lambda=args.l2_inner_loop)
    elif args.algorithm == 'imp_reg_maml':
        algorithm = ImpRMAML_inner_algorithm(
            model=model,
            embedding_model=embedding_model,
            inner_loss_func=loss_func,
            l2_lambda=args.l2_inner_loop,
            device=args.device,
            is_classification=True)


    if args.algorithm == 'imp_reg_maml':
        trainer = Implicit_Gradient_based_algorithm_trainer(
                algorithm=algorithm,
                outer_loss_func=loss_func,
                outer_optimizer=optimizers, 
                writer=writer,
                log_interval=args.log_interval, save_interval=args.save_interval,
                model_type=args.model_type, save_folder=save_folder, outer_loop_grad_norm=args.model_grad_clip)

    else:
        trainer = Gradient_based_algorithm_trainer(
            algorithm=algorithm, outer_loss_func=loss_func,
            outer_optimizer=optimizers, writer=writer,
            log_interval=args.log_interval, save_interval=args.save_interval,
            model_type=args.model_type, save_folder=save_folder, outer_loop_grad_norm=args.model_grad_clip)

    
    if is_training:
        # create train iterators
        train_iterator = iter(dataset['train']) 
        for iter_start in range(1, num_batches['train'], args.val_interval):
            try:
                train_result = trainer.run(train_iterator, is_training=True, 
                    start=iter_start, stop=iter_start+args.val_interval)
            except StopIteration:
                print("Finished training iterations.")
                print(train_result)
                print("Starting final validation.")
    
            # validation
            print("\n\n", "=="*27, "\n Starting validation\n", "=="*27)
            val_result = trainer.run(iter(dataset['val']), is_training=False, meta_val=True, start=iter_start+args.val_interval - 1)
            print(val_result)
            print("\n", "=="*27, "\n Finished validation\n", "=="*27)
            
    else:
        results = trainer.run(iter(dataset['val']), is_training=False, start=0)
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(results)
        name = args.checkpoint[0:args.checkpoint.rfind('.')]
        with open(name + '_eval.pkl', 'wb') as f:
            pickle.dump(results, f)


if __name__ == '__main__':

    def str2bool(arg):
        return arg.lower() == 'true'

    parser = argparse.ArgumentParser(
        description='Model-Agnostic Meta-Learning (MAML)')

    # python main.py -dataset multimodal_few_shot --multimodal_few_shot omniglot miniimagenet cifar bird aircraft --mmaml-model True --num-batches 600000 --output-folder mmaml_5mode_5w1s

    # Algorithm
    parser.add_argument('--algorithm', type=str, help='type of algorithm')

    # parser.add_argument('--mmaml-model', type=str2bool, default=False,
    #     help='gated_conv + ConvGRU')
    # parser.add_argument('--maml-model', type=str2bool, default=False,
    #     help='conv')

    # Model
    parser.add_argument('--hidden-sizes', type=int,
        default=[256, 128, 64, 64], nargs='+',
        help='number of hidden units per layer')
    parser.add_argument('--model-type', type=str, default='gatedconv',
        help='type of the model')
    parser.add_argument('--condition-type', type=str, default='affine',
        choices=['affine', 'sigmoid', 'softmax'],
        help='type of the conditional layers')
    parser.add_argument('--condition-order', type=str, default='low2high',
        help='order of the conditional layers to be used')
    parser.add_argument('--use-max-pool', type=str2bool, default=False,
        help='choose whether to use max pooling with convolutional model')
    parser.add_argument('--num-channels', type=int, default=32,
        help='number of channels in convolutional layers')
    parser.add_argument('--disable-norm', action='store_true',
        help='disable batchnorm after linear layers in a fully connected model')
    parser.add_argument('--bias-transformation-size', type=int, default=0,
        help='size of bias transformation vector that is concatenated with '
             'input')

    # Embedding model
    parser.add_argument('--embedding-type', type=str, default='',
        help='type of the embedding')
    parser.add_argument('--use-label', action='store_true', help='use task.y to create label')
    parser.add_argument('--embedding-hidden-size', type=int, default=128,
        help='number of hidden units per layer in recurrent embedding model')
    parser.add_argument('--embedding-num-layers', type=int, default=2,
        help='number of layers in recurrent embedding model')
    parser.add_argument('--modulation-dims', type=int, nargs='+', default=0,
        help='dimensions of the embeddings')
    parser.add_argument('--conv-embedding', type=str2bool, default=True,
        help='')
    parser.add_argument('--conv-embedding-batch-norm', type=str2bool, default=True,
        help='')
    parser.add_argument('--conv-embedding-avgpool-after-conv', type=str2bool, default=True,
        help='')
    parser.add_argument('--num-conv-embedding-layer', type=int, default=4,
        help='')
    parser.add_argument('--no-rnn-aggregation', type=str2bool, default=True,
        help='')
    parser.add_argument('--embedding-pooling', type=str,
        choices=['avg', 'max'], default='avg', help='')
    parser.add_argument('--linear-before-rnn', action='store_true',
        help='')
    parser.add_argument('--modulation-mat-rank', type=int, default=128,
        help='rank of the modulation matrix before ')
    parser.add_argument('--original-conv', action='store_true', default=False,
        help='Use original MAML implementation')
    parser.add_argument('--modulation-mat-spec-norm', type=float, default=100.,
        help='max singular value for modulation mat ')

    # Randomly sampled embedding vectors
    parser.add_argument('--num-sample-embedding', type=int, default=0,
        help='number of randomly sampled embedding vectors')
    parser.add_argument(
        '--sample-embedding-file', type=str, default='embeddings',
        help='the file name of randomly sampled embedding vectors')
    parser.add_argument(
        '--sample-embedding-file-type', type=str, default='hdf5')

    # Inner loop
    parser.add_argument('--first-order', action='store_true',
        help='use the first-order approximation of MAML')
    parser.add_argument('--fast-lr', type=float, default=0.05,
        help='learning rate for the 1-step gradient update of MAML')
    parser.add_argument('--inner-loop-grad-clip', type=float, default=20.0,
        help='enable gradient clipping in the inner loop')
    parser.add_argument('--inner-loop-soft-clip-slope', type=float, default=0.0,
                        help='slope of soft gradient clipping in the inner loop beyond')
    parser.add_argument('--num-updates', type=int, default=5,
        help='how many update steps in the inner loop')
    parser.add_argument('--l2-inner-loop', type=float, default=0.0,
        help='lambda for inner loop l2 loss')   

    # Optimization
    parser.add_argument('--num-batches-meta-train', type=int, default=60000,
        help='number of batches (meta-train)')
    parser.add_argument('--num-batches-meta-val', type=int, default=100,
        help='number of batches (meta-val)')
    parser.add_argument('--num-batches-meta-test', type=int, default=100,
        help='number of batches (meta-test)')
    parser.add_argument('--meta-batch-size', type=int, default=10,
        help='number of tasks per batch')
    parser.add_argument('--slow-lr', type=float, default=0.001,
        help='learning rate for the global update of MAML')
    parser.add_argument('--embedding-grad-clip', type=float, default=0.0,
        help='')
    parser.add_argument('--model-grad-clip', type=float, default=0.0,
                        help='')
    parser.add_argument('--momentum', action='store_true', default=False,
        help='momentum update')
    parser.add_argument('--gamma-momentum', type=float, default=0.2,
        help='momentum param gamma')

    # Dataset
    parser.add_argument('--dataset', type=str, default='multimodal_few_shot',
        help='which dataset to use')
    # parser.add_argument('--data-root', type=str, default='data',
    #     help='path to store datasets')
    parser.add_argument('--num-train-classes', type=int, default=1100,
        help='how many classes for training')
    parser.add_argument('--num-classes-per-batch', type=int, default=5,
        help='how many classes per task')
    parser.add_argument('--num-train-samples-per-class', type=int, default=1,
        help='how many samples per class for training')
    parser.add_argument('--num-val-samples-per-class-meta-train', type=int, default=5,
        help='how many samples per class for validation (meta train)')
    parser.add_argument('--num-val-samples-per-class-meta-val', type=int, default=15,
        help='how many samples per class for validation (meta val)')
    parser.add_argument('--num-val-samples-per-class-meta-test', type=int, default=15,
        help='how many samples per class for validation (meta test)')
    parser.add_argument('--img-side-len', type=int, default=28,
        help='width and height of the input images')
    parser.add_argument('--input-range', type=float, default=[-5.0, 5.0],
        nargs='+', help='input range of simple functions')
    parser.add_argument('--phase-range', type=float, default=[0, np.pi],
        nargs='+', help='phase range of sinusoids')
    parser.add_argument('--amp-range', type=float, default=[0.1, 5.0],
        nargs='+', help='amp range of sinusoids')
    parser.add_argument('--slope-range', type=float, default=[-3.0, 3.0],
        nargs='+', help='slope range of linear functions')
    parser.add_argument('--intersect-range', type=float, default=[-3.0, 3.0],
        nargs='+', help='intersect range of linear functions')
    parser.add_argument('--noise-std', type=float, default=0.0,
        help='add gaussian noise to mixed functions')
    parser.add_argument('--oracle', action='store_true',
        help='concatenate phase and amp to sinusoid inputs')
    parser.add_argument('--task-oracle', action='store_true',
        help='uses task id for prediction in some models')
    # Combine few-shot learning datasets
    # parser.add_argument('--multimodal_few_shot', type=str,
    #     default=['omniglot', 'cifar', 'miniimagenet', 'doublemnist', 'triplemnist'], 
    #     choices=['omniglot', 'cifar', 'miniimagenet', 'doublemnist', 'triplemnist',
    #              'bird', 'aircraft'], 
    #     nargs='+')
    parser.add_argument('--multimodal_few_shot', type=str,
        default=['omniglot', 'cifar', 'miniimagenet', 'bird', 'aircraft'], 
        choices=['omniglot', 'cifar', 'miniimagenet', 'doublemnist', 'triplemnist',
                 'bird', 'aircraft'], 
        nargs='+')
    parser.add_argument('--common-img-side-len', type=int, default=84)
    parser.add_argument('--common-img-channel', type=int, default=3,
                        help='3 for RGB and 1 for grayscale')
    parser.add_argument('--mix-meta-batch', type=str2bool, default=True)
    parser.add_argument('--mix-mini-batch', type=str2bool, default=False)


    # Miscellaneous
    parser.add_argument('--output-folder', type=str, default='maml',
        help='name of the output folder')
    parser.add_argument('--device', type=str, default='cuda',
        help='set the device (cpu or cuda)')
    parser.add_argument('--device-number', type=str, default='0',
                        help='gpu device number')
    parser.add_argument('--num-workers', type=int, default=4,
        help='how many DataLoader workers to use')
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

    # parser.add_argument('--alternating', action='store_true',
    #     help='') # not set to True in README.md # alternate between the embedding model optimization and model optimization
    # parser.add_argument('--classifier-schedule', type=int, default=10,
    #     help='')
    # parser.add_argument('--embedding-schedule', type=int, default=10,
    #     help='')

    parser.add_argument('--verbose', type=str2bool, default=False,
        help='')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_number
    print('GPU number', os.environ["CUDA_VISIBLE_DEVICES"])

    # Create logs and saves folder if they don't exist
    if not os.path.exists('./train_dir'):
        os.makedirs('./train_dir')

    # Make sure num sample embedding < num sample tasks
    args.num_sample_embedding = min(args.num_sample_embedding, args.num_batches_meta_train)

    # computer embedding dims
    num_gated_conv_layers = 4
    if args.modulation_dims == 0:
        args.modulation_dims = []
        for i in range(num_gated_conv_layers):
            dim = args.num_channels*2**i
            if args.condition_type == 'affine':
                # needs both gamma and beta
                dim *= 2
            args.modulation_dims.append(dim)

    # assert not (args.mmaml_model and args.maml_model)

    # # mmaml model: gated conv + convGRU
    # if args.mmaml_model is True:
    #     # gatedconv + ConvGRU as default
    #     print('Use MMAML')
    #     args.model_type = 'gatedconv'
    #     args.embedding_type = 'ConvGRU'

    # # maml model: conv
    # if args.maml_model is True:
    #     print('Use vanilla MAML')
    #     args.model_type = 'conv'
    #     args.embedding_type = ''

    # Device
    args.device = torch.device(args.device
        if torch.cuda.is_available() else 'cpu')

    # print args
    if args.verbose:
        print('='*10 + ' ARGS ' + '='*10)
        for k, v in sorted(vars(args).items()):
            print('{}: {}'.format(k, v))
        print('='*26)

    main(args)
