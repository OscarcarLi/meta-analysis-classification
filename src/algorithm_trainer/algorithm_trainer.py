import os
import sys
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import sys
import time
import torch
import torch.nn.functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_
import json
import torch.nn as nn

from src.algorithms.grad import quantile_marks, get_grad_norm_from_parameters
from src.algorithm_trainer.utils import *
from src.algorithms.utils import logistic_regression_grad_with_respect_to_w, logistic_regression_mixed_derivatives_with_respect_to_w_then_to_X






class Meta_algorithm_trainer(object):

    def __init__(self, algorithm, optimizer, writer, log_interval, 
        save_folder, grad_clip, label_offset=0, init_global_iteration=0):

        self._algorithm = algorithm
        self._optimizer = optimizer
        self._writer = writer
        self._log_interval = log_interval 
        self._save_folder = save_folder
        self._grad_clip = grad_clip
        self._label_offset = label_offset
        self._global_iteration = init_global_iteration
        self._eps = 0.
        

    def run(self, mt_loader, mt_manager, epoch=None, is_training=True, randomize_query=False):

        if is_training:
            self._algorithm._model.train()
        else:
            self._algorithm._model.eval()

        # loaders and iterators
        mt_iterator = tqdm(enumerate(mt_loader, start=1),
                        leave=False, file=sys.stdout, position=0)
        
        # metrics aggregation
        aggregate = defaultdict(list)
        
        # constants
        n_way = mt_manager.n_way
        n_shot = mt_manager.n_shot
        mt_batch_sz = mt_manager.batch_size
        n_query = mt_manager.n_query
        mt_batch_sz = mt_manager.batch_size        
        print(f"n_way: {n_way}, n_shot: {n_shot}, n_query: {n_query} mt_batch_sz: {mt_batch_sz} randomize_query: {randomize_query}")
        

        for i, mt_batch in mt_iterator:
                    
            # global iterator count
            self._global_iteration += 1
            analysis = (i % self._log_interval == 0)

            # randperm
            if randomize_query and is_training:
                rp = np.random.permutation(2 * n_query * n_way)[:n_query * n_way]
            else:
                rp = None 

            # meta-learning data creation
            mt_batch_x, mt_batch_y = mt_batch
            mt_batch_y = mt_batch_y - self._label_offset
            original_shape = mt_batch_x.shape
            assert len(original_shape) == 5
            # (batch_sz*n_way, n_shot+n_query, channels , height , width)
            mt_batch_x = mt_batch_x.reshape(mt_batch_sz, n_way, *original_shape[-4:])
            # (batch_sz, n_way, n_shot+n_query, channels , height , width)
            shots_x = mt_batch_x[:, :, :n_shot, :, :, :]
            # (batch_sz, n_way, n_shot, channels , height , width)
            if rp is None:
                query_x = mt_batch_x[:, :, n_shot:, :, :, :]
            else:
                query_x = []
                for c in range(n_way):
                    indices = rp[(rp>=(c*2*n_query)) & (rp<((c+1)*2*n_query))] - (c*2*n_query)
                    query_x.append(mt_batch_x[:, c, n_shot + indices, :, :, :])
                query_x = torch.cat(query_x, dim=1)
            # (batch_sz, n_way, n_query, channels , height , width)
            shots_x = shots_x.reshape(mt_batch_sz, -1, *original_shape[-3:])
            # (batch_sz, n_way*n_shot, channels , height , width)
            query_x = query_x.reshape(mt_batch_sz, -1, *original_shape[-3:])
            # (batch_sz, n_way*n_query, channels , height , width)
            shots_y, query_y = get_labels(mt_batch_y, n_way=n_way, 
                n_shot=n_shot, n_query=n_query, batch_sz=mt_batch_sz, rp=rp)
            assert shots_x.shape == (mt_batch_sz, n_way*n_shot, *original_shape[-3:])
            assert query_x.shape == (mt_batch_sz, n_way*n_query, *original_shape[-3:])
            assert shots_y.shape == (mt_batch_sz, n_way*n_shot)
            assert query_y.shape == (mt_batch_sz, n_way*n_query)

            # to cuda
            shots_x = shots_x.cuda()
            query_x = query_x.cuda()
            shots_y = shots_y.cuda()
            query_y = query_y.cuda()
            
            # compute logits and loss on query
            if is_training:
                logits, measurements_trajectory = self._algorithm.inner_loop_adapt(
                    query=query_x, support=shots_x, support_labels=shots_y,
                    n_way=n_way, n_shot=n_shot, n_query=n_query)
            else:
                with torch.no_grad():
                    logits, measurements_trajectory = self._algorithm.inner_loop_adapt(
                    query=query_x, support=shots_x, support_labels=shots_y,
                    n_way=n_way, n_shot=n_shot, n_query=n_query)
                    
            logits = logits.reshape(-1, logits.size(-1))
            query_y = query_y.reshape(-1)
            assert logits.size(0) == query_y.size(0)
            loss = smooth_loss(
                logits, query_y, logits.shape[1], self._eps)
            accu = accuracy(logits, query_y) * 100.

            # metrics accumulation
            aggregate['mt_outer_loss'].append(loss.item())
            aggregate['mt_outer_accu'].append(accu)
            for k in measurements_trajectory:
                aggregate[k].append(measurements_trajectory[k][-1])
            
            # optimizer step
            if is_training:
                self._optimizer.zero_grad()
                loss.backward()
                if self._grad_clip > 0.:
                    clip_grad_norm_(self._algorithm._model.parameters(), 
                        max_norm=self._grad_clip, norm_type='inf')
                self._optimizer.step()

                    
            # logging
            if analysis and is_training:
                metrics = {}
                for name, values in aggregate.items():
                    metrics[name] = np.mean(values)
                self.log_output(epoch, i, metrics)
                aggregate = defaultdict(list)    


        # save model and log tboard for eval
        if is_training and self._save_folder is not None:
            save_name = "chkpt_{0:03d}.pt".format(epoch)
            save_path = os.path.join(self._save_folder, save_name)
            with open(save_path, 'wb') as f:
                torch.save({'model': self._algorithm._model.state_dict(),
                           'optimizer': self._optimizer}, f)


        results = {
            'train_loss_trajectory': {
                'loss': np.mean(aggregate['loss']), 
                'accu': np.mean(aggregate['accu']),
            },
            'test_loss_after': {
                'loss': np.mean(aggregate['mt_outer_loss']),
                'accu': np.mean(aggregate['mt_outer_accu']),
            }
        }
        mean, i95 = (np.mean(aggregate['mt_outer_accu']), 
            1.96 * np.std(aggregate['mt_outer_accu']) / np.sqrt(len(aggregate['mt_outer_accu'])))
        results['val_task_acc'] = "{:.2f} ± {:.2f} %".format(mean, i95) 
    
        return results



    def log_output(self, epoch, iteration,
                metrics_dict):
        if iteration is not None:
            log_array = ['Epoch {} Iteration {}'.format(epoch, iteration)]
        else:
            log_array = ['Epoch {} '.format(epoch)]
        for key in metrics_dict:
            log_array.append(
                '{}: \t{:.2f}'.format(key, metrics_dict[key]))
            if self._writer is not None:
                self._writer.add_scalar(
                    key, metrics_dict[key], self._global_iteration)
        log_array.append(' ') 
        tqdm.write('\n'.join(log_array))







"""
To learn the initialization parameters
"""

class Init_algorithm_trainer(object):

    def __init__(self, algorithm, optimizer, writer, log_interval, 
        save_folder, grad_clip, num_updates_inner_train, num_updates_inner_val,
        label_offset=0, init_global_iteration=0):

        self._algorithm = algorithm
        self._optimizer = optimizer
        self._writer = writer
        self._log_interval = log_interval 
        self._save_folder = save_folder
        self._grad_clip = grad_clip
        self._num_updates_inner_train = num_updates_inner_train
        self._num_updates_inner_val = num_updates_inner_val
        self._label_offset = label_offset
        self._global_iteration = init_global_iteration
        print("Starting tboard logs from iter", self._global_iteration)
        

    def run(self, mt_loader, mt_manager, epoch=None, is_training=True, randomize_query=False):

        # always transductive
        self._algorithm._model.train()

        # loaders and iterators
        mt_iterator = tqdm(enumerate(mt_loader, start=1),
                        leave=False, file=sys.stdout, position=0)
        
        # metrics aggregation
        aggregate = defaultdict(list)
        
        # constants
        n_way = mt_manager.n_way
        n_shot = mt_manager.n_shot
        n_query = mt_manager.n_query
        mt_batch_sz = mt_manager.batch_size        
        print(f"n_way: {n_way}, n_shot: {n_shot}, n_query: {n_query} mt_batch_sz: {mt_batch_sz} randomize_query: {randomize_query}")

        # iterating over tasks
        for i, mt_batch in mt_iterator:

            # set zero grad
            if is_training:
                self._optimizer.zero_grad()
            
            # global iterator count
            self._global_iteration += 1
            analysis = (i % self._log_interval == 0)

            # randperm
            if randomize_query and is_training:
                rp = np.random.permutation(mgr_n_query * n_way)[:n_query * n_way]
            else:
                rp = None 

            # meta-learning data creation
            mt_batch_x, mt_batch_y = mt_batch
            mt_batch_y = mt_batch_y - self._label_offset
            original_shape = mt_batch_x.shape
            assert len(original_shape) == 5
            # (batch_sz*n_way, n_shot+n_query, channels , height , width)
            mt_batch_x = mt_batch_x.reshape(mt_batch_sz, n_way, *original_shape[-4:])
            # (batch_sz, n_way, n_shot+n_query, channels , height , width)
            shots_x = mt_batch_x[:, :, :n_shot, :, :, :]
            # (batch_sz, n_way, n_shot, channels , height , width)
            if rp is None:
                query_x = mt_batch_x[:, :, n_shot:, :, :, :]
            else:
                query_x = []
                for c in range(n_way):
                    indices = rp[(rp>=(c*2*n_query)) & (rp<((c+1)*2*n_query))] - (c*2*n_query)
                    query_x.append(mt_batch_x[:, c, n_shot + indices, :, :, :])
                query_x = torch.cat(query_x, dim=1)
            # (batch_sz, n_way, n_query, channels , height , width)
            shots_x = shots_x.reshape(mt_batch_sz, -1, *original_shape[-3:])
            # (batch_sz, n_way*n_shot, channels , height , width)
            query_x = query_x.reshape(mt_batch_sz, -1, *original_shape[-3:])
            # (batch_sz, n_way*n_query, channels , height , width)
            shots_y, query_y = get_labels(mt_batch_y, n_way=n_way, 
                n_shot=n_shot, n_query=n_query, batch_sz=mt_batch_sz, rp=rp)
            assert shots_x.shape == (mt_batch_sz, n_way*n_shot, *original_shape[-3:])
            assert query_x.shape == (mt_batch_sz, n_way*n_query, *original_shape[-3:])
            assert shots_y.shape == (mt_batch_sz, n_way*n_shot)
            assert query_y.shape == (mt_batch_sz, n_way*n_query)

            # to cuda
            shots_x = shots_x.cuda()
            query_x = query_x.cuda()
            shots_y = shots_y.cuda()
            query_y = query_y.cuda()
            
            for task_id in range(mt_batch_sz):
                # compute outer gradients and populate model grad with it
                # so that we can directly call optimizer.step()
                measurements_trajectory = self._algorithm.inner_loop_adapt(
                    query=query_x[task_id:task_id+1], 
                    query_labels=query_y[task_id:task_id+1], 
                    support=shots_x[task_id:task_id+1],  
                    support_labels=shots_y[task_id:task_id+1],
                    n_way=n_way, n_shot=n_shot, n_query=n_query,
                    num_updates_inner=self._num_updates_inner_train\
                         if is_training else self._num_updates_inner_val)

                # metrics accumulation
                for k in measurements_trajectory:
                    aggregate[k].append(measurements_trajectory[k][-1])
            
            # optimizer step
            if is_training:
                for param in self._algorithm._model.parameters():
                    param.grad /= mt_batch_sz
                if self._grad_clip > 0.:
                    clip_grad_norm_(self._algorithm._model.parameters(), 
                        max_norm=self._grad_clip, norm_type='inf')
                self._optimizer.step()

            # logging
            if analysis and is_training:
                metrics = {}
                for name, values in aggregate.items():
                    metrics[name] = np.mean(values)
                self.log_output(epoch, i, metrics)
                aggregate = defaultdict(list)    

        # save model and log tboard for eval
        if is_training and self._save_folder is not None:
            save_name = "chkpt_{0:03d}.pt".format(epoch)
            save_path = os.path.join(self._save_folder, save_name)
            with open(save_path, 'wb') as f:
                torch.save({'model': self._algorithm._model.state_dict(),
                           'optimizer': self._optimizer}, f)


        results = {
            'train_loss_trajectory': {
                'loss': np.mean(aggregate['loss']), 
                'accu': np.mean(aggregate['accu']),
            },
            'test_loss_after': {
                'loss': np.mean(aggregate['mt_outer_loss']),
                'accu': np.mean(aggregate['mt_outer_accu']),
            }
        }
        mean, i95 = (np.mean(aggregate['mt_outer_accu']), 
            1.96 * np.std(aggregate['mt_outer_accu']) / np.sqrt(len(aggregate['mt_outer_accu'])))
        results['val_task_acc'] = "{:.2f} ± {:.2f} %".format(mean, i95) 
    
        return results



    def log_output(self, epoch, iteration,
                metrics_dict):
        if iteration is not None:
            log_array = ['Epoch {} Iteration {}'.format(epoch, iteration)]
        else:
            log_array = ['Epoch {} '.format(epoch)]
        for key in metrics_dict:
            log_array.append(
                '{}: \t{:.2f}'.format(key, metrics_dict[key]))
            if self._writer is not None:
                self._writer.add_scalar(
                    key, metrics_dict[key], self._global_iteration)
        log_array.append(' ') 
        tqdm.write('\n'.join(log_array))