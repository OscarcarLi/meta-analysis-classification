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

from algorithm_trainer.algorithms.grad import quantile_marks, get_grad_norm_from_parameters
from algorithm_trainer.utils import accuracy
from algorithm_trainer.algorithms.logistic_regression_utils import logistic_regression_grad_with_respect_to_w
from algorithm_trainer.algorithms.logistic_regression_utils import logistic_regression_mixed_derivatives_with_respect_to_w_then_to_X




def update_sum_measurements(sum_measurements, measurements):
    for key in measurements.keys():
        sum_measurements[key] += np.sum(measurements[key])

def update_sum_measurements_trajectory(sum_measurements_trajectory, measurements_trajectory):
    for key in measurements_trajectory:
        sum_measurements_trajectory[key] += np.sum(measurements_trajectory[key], axis=0)

def divide_measurements(measurements, n):
    for key in measurements:
        measurements[key] /= n
    return measurements

def average_measurements(measurements):
    # measurements is a dictionary from
    # measurement's name to a list of measurements over the batch of tasks
    avg_measurements = {}
    for key in measurements.keys():
        avg_measurements[key] = torch.mean(measurements[key]).item()
    return avg_measurements

def average_measurements_trajectory(measurements_trajectory):
    avg_measurements_trajectory = {}
    for key in measurements_trajectory:
        avg_measurements_trajectory[key] = np.mean(measurements_trajectory[key], axis=0)
    return avg_measurements_trajectory

def standard_deviation_measurement(measurements):
    std_measurements = {}
    for key in measurements.keys():
        std_measurements[key] = torch.std(measurements[key]).item()
    return std_measurements



class LR_algorithm_trainer(object):

    def __init__(self, algorithm, outer_loss_func, outer_optimizer,
            writer, log_interval, save_interval, model_type, save_folder, outer_loop_grad_norm,
            grad_clip=0., hessian_inverse=False):

        self._algorithm = algorithm
        self._outer_loss_func = outer_loss_func
        self._outer_optimizer = outer_optimizer
        self._writer = writer
        self._log_interval = log_interval # at log_interval will do gradient analysis
        self._save_interval = save_interval
        self._model_type = model_type
        self._save_folder = save_folder
        self._grad_clip = grad_clip
        self._hessian_inverse = hessian_inverse


    def run(self, dataset_iterator, is_training=False, meta_val=False, start=1, stop=1):

        if is_training:
            self._algorithm._model.train()
        else:
            self._algorithm._model.eval()
            val_task_acc = []

        # looping through the entire meta_dataset once
        sum_train_measurements_trajectory_over_meta_set = defaultdict(float)
        sum_test_measurements_after_adapt_over_meta_set = defaultdict(float)
        n_tasks = 0

        iterator = tqdm(enumerate(dataset_iterator, start=start if is_training else 1),
                        leave=False, file=sys.stdout, initial=start, position=0)
        for i, (train_task_batch, test_task_batch) in iterator:
            if is_training and i == stop:
                return {'train_loss_trajectory': divide_measurements(
                    sum_train_measurements_trajectory_over_meta_set, n=n_tasks),
                    'test_loss_after': divide_measurements(
                        sum_test_measurements_after_adapt_over_meta_set, n=n_tasks)}

    
            # _meta_dataset yields data iteration
            train_measurements_trajectory_over_batch = defaultdict(list)
            # test_measurements_before_adapt_over_batch = defaultdict(list)
            test_measurements_after_adapt_over_batch = defaultdict(list)
            analysis = (i % self._log_interval == 0)


            batch_size = len(train_task_batch)

            if is_training:
                self._outer_optimizer.zero_grad()

            for train_task, test_task in zip(train_task_batch, test_task_batch):
                # adapt according train_task
                adapted_params, features_train, modulation_train, train_hessian_inv_multiply,\
                train_mixed_partials_left_multiply, train_measurements_trajectory, info_dict = \
                        self._algorithm.inner_loop_adapt(
                            train_task, hessian_inverse=self._hessian_inverse, iter=i) 
                # if hessian_inverse is True then train_hessian is in face train_hessian_inv
                
                for key, measurements in train_measurements_trajectory.items():
                    train_measurements_trajectory_over_batch[key].append(measurements)

                if is_training:
                    features_test = self._algorithm._model(batch=test_task.x, 
                        modulation=modulation_train)
                else:
                    with torch.no_grad():
                        features_test = self._algorithm._model(batch=test_task.x, 
                            modulation=modulation_train)

                if isinstance(self._algorithm._model, torch.nn.DataParallel):
                    scale = self._algorithm._model.module.scale
                else:
                    scale = self._algorithm._model.scale

                test_pred_after_adapt = scale * F.linear(
                    features_test, weight=adapted_params)
                test_loss_after_adapt = self._outer_loss_func(
                    test_pred_after_adapt, test_task.y)
                
                test_measurements_after_adapt_over_batch['loss'].append(
                    test_loss_after_adapt.item())
                test_loss_after_adapt /= batch_size 
                # now we are doing this one by one so need to divide individually

                if self._algorithm.is_classification:
                    task_accuracy = accuracy(test_pred_after_adapt, test_task.y)
                    test_measurements_after_adapt_over_batch['accu'].append(
                        task_accuracy
                    )
                    if not is_training:
                        val_task_acc.append(task_accuracy * 100.)
                
                if is_training:
                    X_test = features_test.detach().cpu().numpy()
                    y_test = (test_task.y).cpu().numpy()
                    w = adapted_params.detach().cpu().numpy()
                    test_grad_w = logistic_regression_grad_with_respect_to_w(X_test, y_test,
                        scale.detach().cpu().numpy() * w)

                    train_hessian_inv_test_grad = train_hessian_inv_multiply(
                        test_grad_w)
                    test_grad_features_train = - train_mixed_partials_left_multiply(
                        train_hessian_inv_test_grad)
                    test_grad_features_train = test_grad_features_train.reshape(
                        features_train.shape)

                    features_train.backward(gradient=(
                        torch.tensor(test_grad_features_train,
                        device=self._algorithm._device) / batch_size),
                        retain_graph=True,
                        create_graph=False)
                    test_loss_after_adapt.backward(retain_graph=False, create_graph=False)

            update_sum_measurements_trajectory(sum_train_measurements_trajectory_over_meta_set,
                                               train_measurements_trajectory_over_batch)
            update_sum_measurements(sum_test_measurements_after_adapt_over_meta_set,
                                    test_measurements_after_adapt_over_batch)
            n_tasks += batch_size

            if is_training:
                outer_model_grad_norm_before_clip = get_grad_norm_from_parameters(
                    self._algorithm._model.parameters())
                self._writer.add_scalar('outer_grad/model_norm/before_clip',
                    outer_model_grad_norm_before_clip, i)
                if self._grad_clip > 0.:
                    clip_grad_norm_(
                        self._algorithm._model.parameters(), self._grad_clip)
                self._outer_optimizer.step()

            if analysis and is_training:
                self.log_output(i,
                    train_measurements_trajectory_over_batch,
                    test_measurements_after_adapt_over_batch,
                    write_tensorboard=is_training)

            # Save model
            if (i % self._save_interval == 0 or i ==1) and is_training:
                save_name = '{0}_{1:04d}.pt'.format(self._model_type, i)
                save_path = os.path.join(self._save_folder, save_name)
                with open(save_path, 'wb') as f:
                    torch.save(self._algorithm.state_dict(), f)
        
        results = {'train_loss_trajectory': divide_measurements(
            sum_train_measurements_trajectory_over_meta_set, n=n_tasks),
               'test_loss_after': divide_measurements(
                   sum_test_measurements_after_adapt_over_meta_set, n=n_tasks)}
        
        if (not is_training) and meta_val:
            self.log_output(
                start,
                results['train_loss_trajectory'],
                results['test_loss_after'],
                write_tensorboard=True, meta_val=True)

        if not is_training:
            mean, i95 = (np.mean(val_task_acc), 
                1.96 * np.std(val_task_acc) / np.sqrt(len(val_task_acc)))
            results['val_task_acc'] = "{:.2f} ± {:.2f} %".format(mean, i95) 
        
        return results


    def log_output(self, iteration,
                train_measurements_trajectory_over_batch,
                test_measurements_after_adapt_over_batch,
                write_tensorboard=False, meta_val=False):

        log_array = ['Iteration: {}'.format(iteration)]
        key_list = ['loss']
        if self._algorithm.is_classification: key_list.append('accu')
        for key in key_list:
            if not meta_val:
                avg_train_trajectory = np.mean(train_measurements_trajectory_over_batch[key], axis=0)
                avg_test_after = np.mean(test_measurements_after_adapt_over_batch[key])
                avg_train_after = avg_train_trajectory[-1]
            else:
                avg_train_trajectory = train_measurements_trajectory_over_batch[key]
                avg_test_after = test_measurements_after_adapt_over_batch[key]
                avg_train_after = avg_train_trajectory[-1]

            if key == 'accu':
                log_array.append('train {} after: \t{:.2f}%'.format(key, 100 * avg_train_after))
                log_array.append('test {} after: \t{:.2f}%'.format(key, 100 * avg_test_after))
            else:
                log_array.append('train {} after: \t{:.3f}'.format(key, avg_train_after))
                log_array.append('test {} after: \t{:.3f}'.format(key, avg_test_after))

            if write_tensorboard:
                if meta_val:
                    self._writer.add_scalar('meta_val/train_{}_post'.format(key),
                                                avg_train_trajectory[-1],
                                                iteration)
                    self._writer.add_scalar('meta_val/test_{}_post'.format(key), avg_test_after, iteration)
                else:
                    self._writer.add_scalar('meta_train/train_{}_post'.format(key),
                                                avg_train_trajectory[-1],
                                                iteration)
                    self._writer.add_scalar('meta_train/test_{}_post'.format(key), avg_test_after, iteration)

            log_array.append(' ') 
        if not meta_val:
            tqdm.write('\n'.join(log_array))

    def write_gradient_info_to_board(self, iteration,
            grad_norm_by_step_over_batch,
            grad_quantiles_by_step_over_batch):
        avg_grad_norm_by_step = np.mean(grad_norm_by_step_over_batch, axis=0)
        avg_grad_quantiles_by_step = defaultdict(list)
        for step in grad_quantiles_by_step_over_batch.keys():
            avg_grad_quantiles_by_step[step] =\
                np.mean(grad_quantiles_by_step_over_batch[step],
                        axis=0)
        for step_i, grad_norm in enumerate(avg_grad_norm_by_step, start=1):
            self._writer.add_scalar(
                'inner_grad/norm/{}-inner gradient step'.format(step_i), 
                grad_norm, iteration)
        for step_i, quantiles in avg_grad_quantiles_by_step.items():
            for qm, quantile_value in zip(quantile_marks, quantiles):
                self._writer.add_scalar(
                    'inner_grad/quantile/{}-inner gradient/{} quantile'.format(
                        step_i, qm), quantile_value, iteration)

    
    def write_embeddings_output_to_board(self, embeddings_output, metadata, iteration):
        embeddings_output = [torch.stack(x, dim=0).squeeze(1) for x in embeddings_output]
        embeddings_output = torch.stack(embeddings_output, dim=0)
        for layer in range(embeddings_output.size(1)):
            self._writer.add_embedding(
                embeddings_output[:, layer, :],
                metadata=metadata,
                tag=f'embedding_layer_{layer}',
                global_step=iteration
            )




"""
Note: for metaoptnet the trainer passes a batch of tasks to inner loop 
as opposed to a single task. This optimization is to benefit from batch mm type operations,
so as to offset the time take by qp solver.
"""

class Generic_algorithm_trainer(object):

    def __init__(self, algorithm, outer_loss_func, outer_optimizer,
            writer, log_interval, save_interval, save_folder, model_type, 
            outer_loop_grad_norm, grad_clip=0., optimizer_update_interval=1):

        self._algorithm = algorithm
        self._outer_loss_func = outer_loss_func
        self._outer_optimizer = outer_optimizer
        self._writer = writer
        self._log_interval = log_interval 
        # at log_interval will do gradient analysis
        self._save_interval = save_interval
        self._save_folder = save_folder
        self._grad_clip = grad_clip
        self._model_type = model_type 
        self._optimizer_update_interval = optimizer_update_interval
        

    def run(self, dataset_iterator, dataset_manager, is_training=False, meta_val=False, start=1, stop=1, fixed_batches=None):

        if is_training:
            self._algorithm._model.train()
        else:
            self._algorithm._model.eval()
            val_task_acc = []

        # looping through the entire meta_dataset once
        sum_train_measurements_trajectory_over_meta_set = defaultdict(float)
        sum_test_measurements_after_adapt_over_meta_set = defaultdict(float)
        n_task_batches = 0

        n_way = dataset_manager.n_way
        n_shot = dataset_manager.n_shot
        n_query = dataset_manager.n_query
        batch_sz = dataset_manager.batch_size
        print(f"n_way: {n_way}, n_shot: {n_shot}, n_query: {n_query}, batch_sz: {batch_sz}")

        if fixed_batches is None:
            iterator = tqdm(enumerate(dataset_iterator, start=start if is_training else 1),
                            leave=False, file=sys.stdout, initial=start, position=0)
        else:
            iterator = tqdm(enumerate(zip(*fixed_batches), start=start if is_training else 1),
                            leave=False, file=sys.stdout, initial=start, position=0)


        for i, batch in iterator:

            if fixed_batches is None:

                ############## covariates #############
                x_batch, y_batch = batch
                original_shape = x_batch.shape
                assert len(original_shape) == 5
                # (batch_sz*n_way, n_shot+n_query, channels , height , width)
                x_batch = x_batch.reshape(batch_sz, n_way, *original_shape[-4:])
                # (batch_sz, n_way, n_shot+n_query, channels , height , width)
                shots_x = x_batch[:, :, :n_shot, :, :, :]
                # (batch_sz, n_way, n_shot, channels , height , width)
                query_x = x_batch[:, :, n_shot:, :, :, :]
                # (batch_sz, n_way, n_query, channels , height , width)
                shots_x = shots_x.reshape(batch_sz, -1, *original_shape[-3:])
                # (batch_sz, n_way*n_shot, channels , height , width)
                query_x = query_x.reshape(batch_sz, -1, *original_shape[-3:])
                # (batch_sz, n_way*n_query, channels , height , width)

                ############## labels #############
                shots_y, query_y = self.get_labels(y_batch, n_way=n_way, 
                    n_shot=n_shot, n_query=n_query, batch_sz=batch_sz)
        
            else:
                ## to be used only in eval mode
                assert is_training is False
                shots_x, shots_y, query_x, query_y = batch
                uniq_classes = np.unique(shots_y)
                assert all(uniq_classes == np.unique(query_y))
                conversion_dict = {v:k for k, v in enumerate(uniq_classes)}
                # convert labels
                for uniq_class in uniq_classes: 
                    shots_y[shots_y==uniq_class] = conversion_dict[uniq_class]
                    query_y[query_y==uniq_class] = conversion_dict[uniq_class]
                    
                batch_sz = 1 # hack, since batches are returned by fine tune method in classical trainer
                shots_x = shots_x.unsqueeze(dim=0)
                shots_y = shots_y.unsqueeze(dim=0)
                query_x = query_x.unsqueeze(dim=0)
                query_y = query_y.unsqueeze(dim=0)
                original_shape = shots_x.shape
                
            
            # sanity checks
            assert shots_x.shape == (batch_sz, n_way*n_shot, *original_shape[-3:])
            assert query_x.shape == (batch_sz, n_way*n_query, *original_shape[-3:])
            assert shots_y.shape == (batch_sz, n_way*n_shot)
            assert query_y.shape == (batch_sz, n_way*n_query)

            # move labels and covariates to cuda
            shots_x = shots_x.cuda()
            query_x = query_x.cuda()
            shots_y = shots_y.cuda()
            query_y = query_y.cuda()


            if is_training and i == stop:
                return {'train_loss_trajectory': divide_measurements(
                    sum_train_measurements_trajectory_over_meta_set, n=n_task_batches),
                    'test_loss_after': divide_measurements(
                        sum_test_measurements_after_adapt_over_meta_set, n=n_task_batches)}
            
            if is_training and (i % self._optimizer_update_interval == 0):
                self._outer_optimizer.zero_grad()

            if is_training:    
                logits, measurements_trajectory = self._algorithm.inner_loop_adapt(
                    query=query_x, support=shots_x, 
                    support_labels=shots_y)
                assert len(set(shots_y)) == len(set(query_y))
            else:
                with torch.no_grad():
                    logits, measurements_trajectory = self._algorithm.inner_loop_adapt(
                        query=query_x, support=shots_x, 
                        support_labels=shots_y)
                    assert len(set(shots_y)) == len(set(query_y))


            if isinstance(self._algorithm._model, torch.nn.DataParallel):
                scale = self._algorithm._model.module.scale
            else:
                scale = self._algorithm._model.scale

            # reshape logits
            logits = scale * logits.reshape(-1, logits.size(-1))
            query_y = query_y.reshape(-1)
            assert logits.size(0) == query_y.size(0)
            analysis = (i % self._log_interval == 0)

            # compute loss abd accu
            test_loss_after_adapt = self._outer_loss_func(logits, query_y)
            test_accu_after_adapt = accuracy(logits, query_y)
            if not is_training:
                val_task_acc.append(test_accu_after_adapt * 100.)

            if is_training:
                test_loss_after_adapt.backward()
        
            # metrics
            train_measurements_trajectory_over_batch = {
                k:np.array([v]) for k,v in measurements_trajectory.items()
            }
            test_measurements_after_adapt_over_batch = {
                'loss': np.array([test_loss_after_adapt.item()]) , 
                'accu': np.array([test_accu_after_adapt])
            }
                
            update_sum_measurements(sum_test_measurements_after_adapt_over_meta_set,
                                    test_measurements_after_adapt_over_batch)
            update_sum_measurements_trajectory(sum_train_measurements_trajectory_over_meta_set,
                                               train_measurements_trajectory_over_batch)
            
            n_task_batches += 1

            if is_training and (i % self._optimizer_update_interval == 0):
                outer_model_grad_norm_before_clip = get_grad_norm_from_parameters(
                    self._algorithm._model.parameters())
                self._writer.add_scalar(
                    'outer_grad/model_norm/before_clip', outer_model_grad_norm_before_clip, i)
                if self._grad_clip > 0.:
                    clip_grad_norm_(
                        self._algorithm._model.parameters(), self._grad_clip)
                self._outer_optimizer.step()

            # logging
            if analysis and is_training:
                self.log_output(i,
                    train_measurements_trajectory_over_batch,
                    test_measurements_after_adapt_over_batch,
                    write_tensorboard=is_training)


            # Save model
            if (i % self._save_interval == 0 or i ==1) and is_training:
                save_name = '{0}_{1:04d}.pt'.format(self._model_type, i)
                save_path = os.path.join(self._save_folder, save_name)
                with open(save_path, 'wb') as f:
                    torch.save(self._algorithm.state_dict(), f)
        
        results = {
            'train_loss_trajectory': divide_measurements(
                sum_train_measurements_trajectory_over_meta_set, n=n_task_batches),
            'test_loss_after': divide_measurements(
                sum_test_measurements_after_adapt_over_meta_set, n=n_task_batches)
        }
        
        if (not is_training) and meta_val:
            self.log_output(
                start,
                results['train_loss_trajectory'],
                results['test_loss_after'],
                write_tensorboard=True, meta_val=True)

        if not is_training:
            mean, i95 = (np.mean(val_task_acc), 
                1.96 * np.std(val_task_acc) / np.sqrt(len(val_task_acc)))
            results['val_task_acc'] = "{:.2f} ± {:.2f} %".format(mean, i95) 
        
        return results


    def get_labels(self, y_batch, n_way, n_shot, n_query, batch_sz):
        # original y_batch: (batch_sz*n_way, n_shot+n_query)
        y_batch = y_batch.reshape(batch_sz, n_way, -1)
        # batch_sz, n_way, n_shot+n_query
        
        for i in range(y_batch.shape[0]):
            uniq_classes = np.unique(y_batch[i, :, :])
            conversion_dict = {v:k for k, v in enumerate(uniq_classes)}
            # convert labels
            for uniq_class in uniq_classes: 
                y_batch[i, y_batch[i]==uniq_class] = conversion_dict[uniq_class]
            
        shots_y = y_batch[:, :, :n_shot]
        query_y = y_batch[:, :, n_shot:]
        shots_y = shots_y.reshape(batch_sz, -1)
        query_y = query_y.reshape(batch_sz, -1)
        return shots_y, query_y
        

    def log_output(self, iteration,
                train_measurements_trajectory_over_batch,
                test_measurements_after_adapt_over_batch,
                write_tensorboard=False, meta_val=False):

        log_array = ['Iteration: {}'.format(iteration)]
        key_list = ['loss', 'accu']
        for key in key_list:
            if not meta_val:
                avg_train_trajectory = np.mean(train_measurements_trajectory_over_batch[key], axis=0)
                avg_test_after = np.mean(test_measurements_after_adapt_over_batch[key])
                avg_train_after = avg_train_trajectory[-1]
            else:
                avg_train_trajectory = train_measurements_trajectory_over_batch[key]
                avg_test_after = test_measurements_after_adapt_over_batch[key]
                avg_train_after = avg_train_trajectory[-1]

            if 'accu' in key:
                log_array.append('train {} after: \t{:.2f}%'.format(key, 100 * avg_train_after))
                log_array.append('test {} after: \t{:.2f}%'.format(key, 100 * avg_test_after))
            else:
                log_array.append('train {} after: \t{:.3f}'.format(key, avg_train_after))
                log_array.append('test {} after: \t{:.3f}'.format(key, avg_test_after))

            if write_tensorboard:
                if meta_val:
                    self._writer.add_scalar('meta_val/train_{}_post'.format(key),
                                                avg_train_trajectory[-1],
                                                iteration)
                    self._writer.add_scalar('meta_val/test_{}_post'.format(key), avg_test_after, iteration)
                else:
                    self._writer.add_scalar('meta_train/train_{}_post'.format(key),
                                                avg_train_trajectory[-1],
                                                iteration)
                    self._writer.add_scalar('meta_train/test_{}_post'.format(key), avg_test_after, iteration)

            log_array.append(' ') 
        if not meta_val:
            tqdm.write('\n'.join(log_array))

    def write_gradient_info_to_board(self, iteration,
                                     grad_norm_by_step_over_batch,
                                     grad_quantiles_by_step_over_batch):
        avg_grad_norm_by_step = np.mean(grad_norm_by_step_over_batch, axis=0)
        avg_grad_quantiles_by_step = defaultdict(list)
        for step in grad_quantiles_by_step_over_batch.keys():
            avg_grad_quantiles_by_step[step] =\
                np.mean(grad_quantiles_by_step_over_batch[step],
                        axis=0)
        for step_i, grad_norm in enumerate(avg_grad_norm_by_step, start=1):
            self._writer.add_scalar(
                'inner_grad/norm/{}-inner gradient step'.format(step_i), grad_norm, iteration)
        for step_i, quantiles in avg_grad_quantiles_by_step.items():
            for qm, quantile_value in zip(quantile_marks, quantiles):
                self._writer.add_scalar(
                    'inner_grad/quantile/{}-inner gradient/{} quantile'.format(step_i, qm),
                     quantile_value, iteration)

    
    def write_embeddings_output_to_board(self, embeddings_output, metadata, iteration):
        embeddings_output = [torch.stack(x, dim=0).squeeze(1) for x in embeddings_output]
        embeddings_output = torch.stack(embeddings_output, dim=0)
        for layer in range(embeddings_output.size(1)):
            self._writer.add_embedding(
                embeddings_output[:, layer, :],
                metadata=metadata,
                tag=f'embedding_layer_{layer}',
                global_step=iteration
            )




"""
To train model on all classes together
"""

class Classical_algorithm_trainer(object):

    def __init__(self, model, loss_func, optimizer, writer,
        log_interval, save_folder, grad_norm):

        self._model = model
        self._loss_func = loss_func
        self._optimizer = optimizer
        self._writer = writer
        self._log_interval = log_interval 
        self._save_folder = save_folder
        self._grad_norm = grad_norm


    def run(self, dataset_iterator, epoch=None, is_training=True):

        if is_training:
            self._model.train()
        else:
            self._model.eval()
         
        iterator = tqdm(enumerate(dataset_iterator, start=1),
                        leave=False, file=sys.stdout, position=0)
        agg_loss = []
        agg_accu = []
        
        for i, batch in iterator:

            analysis = (i % self._log_interval == 0)
            batch_size = len(batch)
            batch_x, batch_y = batch
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()

            
            logits = self._model(batch_x)
            loss = self._loss_func(logits, batch_y)
            accu = accuracy(logits, batch_y)
            
            if is_training:
                self._optimizer.zero_grad()
                loss.backward()
                if self._grad_norm > 0.:
                    clip_grad_norm_(self._model.parameters(), self._grad_norm)
                self._optimizer.step()
            
            agg_loss.append(loss.data.item())
            agg_accu.append(accu)
            
            # logging
            if analysis and is_training:
                self.log_output(epoch, i,
                    {"train_loss":  np.mean(agg_loss),
                        "train_acc": np.mean(agg_accu) * 100.})
                agg_loss = []
                agg_accu = []

        # save model and log tboard for eval
        if is_training:
            save_name = "classical_{0}_{1:03d}.pt".format('resnet', epoch)
            save_path = os.path.join(self._save_folder, save_name)
            with open(save_path, 'wb') as f:
                torch.save({'model': self._model.state_dict()}, f)

        else:
            self.log_output(epoch, None,
                {"val_loss":  np.mean(agg_loss),
                    "val_acc": np.mean(agg_accu) * 100.})    



    def fine_tune(self, dataset_iterator, dataset_manager, label_offset=0, n_fine_tune_epochs=1):

        self._model.train()

        n_way = dataset_manager.n_way
        n_shot = dataset_manager.n_shot
        n_query = dataset_manager.n_query
        batch_sz = dataset_manager.batch_size
        print(f"n_way: {n_way}, n_shot: {n_shot}, n_query: {n_query}, batch_sz: {batch_sz}")

        all_val_tasks_shots_x = []
        all_val_tasks_shots_y = []
        all_val_tasks_query_x = []
        all_val_tasks_query_y = []
        
        print("parsing val dataset once ... ")
        for batch in tqdm(dataset_iterator, total=len(dataset_iterator)):

            ############## covariates #############
            x_batch, y_batch = batch
            original_shape = x_batch.shape
            assert len(original_shape) == 5
            # (batch_sz*n_way, n_shot+n_query, channels , height , width)
            x_batch = x_batch.reshape(batch_sz, n_way, *original_shape[-4:])
            # (batch_sz, n_way, n_shot+n_query, channels , height , width)
            shots_x = x_batch[:, :, :n_shot, :, :, :]
            # (batch_sz, n_way, n_shot, channels , height , width)
            query_x = x_batch[:, :, n_shot:, :, :, :]
            # (batch_sz, n_way, n_query, channels , height , width)
            shots_x = shots_x.reshape(batch_sz, -1, *original_shape[-3:])
            # (batch_sz, n_way*n_shot, channels , height , width)
            query_x = query_x.reshape(batch_sz, -1, *original_shape[-3:])
            # (batch_sz, n_way*n_query, channels , height , width)
            assert shots_x.shape == (batch_sz, n_way*n_shot, *original_shape[-3:])
            assert query_x.shape == (batch_sz, n_way*n_query, *original_shape[-3:])


            ############## labels #############
            y_batch = y_batch.reshape(batch_sz, n_way, -1)
            # batch_sz, n_way, n_shot+n_query
            shots_y = y_batch[:, :, :n_shot].reshape(batch_sz, -1)
            query_y = y_batch[:, :, n_shot:].reshape(batch_sz, -1)
        
            ### accumulate samples across tasks ###
            all_val_tasks_shots_x.append(shots_x)
            all_val_tasks_shots_y.append(shots_y)
            all_val_tasks_query_x.append(query_x)
            all_val_tasks_query_y.append(query_y)


        ############## concatenate samples from all tasks #############
        all_val_tasks_shots_x = torch.cat(all_val_tasks_shots_x, dim=0)
        all_val_tasks_shots_y = torch.cat(all_val_tasks_shots_y, dim=0)
        all_val_tasks_query_x = torch.cat(all_val_tasks_query_x, dim=0)
        all_val_tasks_query_y = torch.cat(all_val_tasks_query_y, dim=0)

        #### fix label offset before fine tuning ####
        """This is mainly because at validation time the samples would be 
        labelled using their flobal values: [64:79] for mini-imagenet for eg.,
        this nneds to be reduced by 64 to get labels from 0 to 15.
        """
        all_val_tasks_shots_y -= label_offset
        all_val_tasks_query_y -= label_offset

        print("all_val_tasks_shots_x", all_val_tasks_shots_x.shape)
        print("all_val_tasks_shots_y", all_val_tasks_shots_y.shape)
        print("all_val_tasks_query_x", all_val_tasks_query_x.shape)
        print("all_val_tasks_query_y", all_val_tasks_query_y.shape)

        ## begin fine tuning ##
        epoch = 0
        for _ in range(n_fine_tune_epochs):
            
            epoch +=1 
            iterator = tqdm(enumerate(zip(all_val_tasks_shots_x, all_val_tasks_shots_y), start=1),
                            leave=False, file=sys.stdout, position=0)

            agg_loss = []
            agg_accu = []
                     
            for i, (shots_x, shots_y) in iterator:

                analysis = (i % self._log_interval == 0)
                batch_size = len(shots_x)
                shots_x = shots_x.reshape(-1, *shots_x.shape[-3:]) 
                shots_y = shots_y.reshape(-1)
                batch_x = shots_x.cuda()
                batch_y = shots_y.cuda()
                
                logits = self._model(batch_x)
                loss = self._loss_func(logits, batch_y)
                accu = accuracy(logits, batch_y)
                
                self._optimizer.zero_grad()
                loss.backward()
                if self._grad_norm > 0.:
                    clip_grad_norm_(self._model.parameters(), self._grad_norm)
                self._optimizer.step()
                
                agg_loss.append(loss.data.item())
                agg_accu.append(accu)
                
                # logging
                if analysis:
                    self.log_output(epoch, i,
                        {"train_loss":  np.mean(agg_loss),
                            "train_acc": np.mean(agg_accu) * 100.}, write_tensorboard=False)
                    agg_loss = []
                    agg_accu = []

        return self._model, (all_val_tasks_shots_x, all_val_tasks_shots_y, all_val_tasks_query_x, all_val_tasks_query_y)
        

    def log_output(self, epoch, iteration,
                metrics_dict, write_tensorboard=True):
        if iteration is not None:
            log_array = ['Epoch {} Iteration {}'.format(epoch, iteration)]
        else:
            log_array = ['Epoch {} '.format(epoch)]
        for key in metrics_dict:
            log_array.append(
                '{}: \t{:.2f}'.format(key, metrics_dict[key]))
            if write_tensorboard:
                self._writer.add_scalar(
                    key, metrics_dict[key], iteration)
            log_array.append(' ') 
        tqdm.write('\n'.join(log_array))




"""Use a generic feature backbone but adapt it using 
some auxiliary adaptation strategy.
"""

class Generic_adaptation_trainer(object):

    def __init__(self, algorithm, aux_objective, outer_loss_func, outer_optimizer,
            writer, log_interval, model_type, grad_clip=0., n_aux_objective_steps=5):

        self._algorithm = algorithm
        self._aux_objective = aux_objective
        self._outer_loss_func = outer_loss_func
        self._outer_optimizer = outer_optimizer
        self._writer = writer
        self._log_interval = log_interval 
        # at log_interval will do gradient analysis
        self._grad_clip = grad_clip
        self._model_type = model_type
        self._n_aux_objective_steps = n_aux_objective_steps 

        


    def optimize_auxiliary_obj(self, shots_x, shots_y):
        for _ in range (self._n_aux_objective_steps):
            self._outer_optimizer.zero_grad()
            orig_shots_shape = shots_x.shape
            features_x = self._algorithm._model(
                shots_x.reshape(-1, *orig_shots_shape[2:])).reshape(*orig_shots_shape[:2], -1)
            aux_loss = self._aux_objective(features_x, shots_y)
            aux_loss.backward()
            if self._grad_clip > 0.:
                clip_grad_norm_(self._algorithm._model.parameters(), self._grad_clip)
            self._outer_optimizer.step()
        return aux_loss.item()
        

    def run(self, dataset_iterator, dataset_manager, is_training=False, meta_val=False, start=1, stop=1):

        val_task_acc = []

        # looping through the entire meta_dataset once
        sum_train_measurements_trajectory_over_meta_set = defaultdict(float)
        sum_test_measurements_after_adapt_over_meta_set = defaultdict(float)
        n_task_batches = 0

        # meta-learning task configurations
        n_way = dataset_manager.n_way
        n_shot = dataset_manager.n_shot
        n_query = dataset_manager.n_query
        batch_sz = dataset_manager.batch_size
        print(f"n_way: {n_way}, n_shot: {n_shot}, n_query: {n_query}, batch_sz: {batch_sz}")

        # iterator
        iterator = tqdm(enumerate(dataset_iterator, start=start if is_training else 1),
                        leave=False, file=sys.stdout, initial=start, position=0)
        
        for i, batch in iterator:
        
            ############## covariates #############
            x_batch, y_batch = batch
            original_shape = x_batch.shape
            assert len(original_shape) == 5
            # (batch_sz*n_way, n_shot+n_query, channels , height , width)
            x_batch = x_batch.reshape(batch_sz, n_way, *original_shape[-4:])
            # (batch_sz, n_way, n_shot+n_query, channels , height , width)
            shots_x = x_batch[:, :, :n_shot, :, :, :]
            # (batch_sz, n_way, n_shot, channels , height , width)
            query_x = x_batch[:, :, n_shot:, :, :, :]
            # (batch_sz, n_way, n_query, channels , height , width)
            shots_x = shots_x.reshape(batch_sz, -1, *original_shape[-3:])
            # (batch_sz, n_way*n_shot, channels , height , width)
            query_x = query_x.reshape(batch_sz, -1, *original_shape[-3:])
            # (batch_sz, n_way*n_query, channels , height , width)

            ############## labels #############
            shots_y, query_y = self.get_labels(y_batch, n_way=n_way, 
                n_shot=n_shot, n_query=n_query, batch_sz=batch_sz)
    

            # sanity checks
            assert shots_x.shape == (batch_sz, n_way*n_shot, *original_shape[-3:])
            assert query_x.shape == (batch_sz, n_way*n_query, *original_shape[-3:])
            assert shots_y.shape == (batch_sz, n_way*n_shot)
            assert query_y.shape == (batch_sz, n_way*n_query)

            # move labels and covariates to cuda
            shots_x = shots_x.cuda()
            query_x = query_x.cuda()
            shots_y = shots_y.cuda()
            query_y = query_y.cuda()


            # cpy model state dict and optimize model on a specific objective
            if self._aux_objective is not None:
                original_state_dict = self._algorithm._model.state_dict()
                self._algorithm._model.train()
                aux_loss = self.optimize_auxiliary_obj(shots_x, shots_y)
            self._algorithm._model.eval()
        
            # forward pass on updated model
            with torch.no_grad():
                logits, measurements_trajectory = self._algorithm.inner_loop_adapt(
                    query=query_x, support=shots_x, 
                    support_labels=shots_y)
                assert len(set(shots_y)) == len(set(query_y))
            if isinstance(self._algorithm._model, torch.nn.DataParallel):
                scale = self._algorithm._model.module.scale
            else:
                scale = self._algorithm._model.scale

            # reshape logits
            logits = scale * logits.reshape(-1, logits.size(-1))
            query_y = query_y.reshape(-1)
            assert logits.size(0) == query_y.size(0)
            analysis = (i % self._log_interval == 0)

            # reinstate original model for the next task
            if self._aux_objective is not None:
                self._algorithm._model.load_state_dict(original_state_dict)

            # compute loss and accu
            test_loss_after_adapt = self._outer_loss_func(logits, query_y)
            test_accu_after_adapt = accuracy(logits, query_y)
            if not is_training:
                val_task_acc.append(test_accu_after_adapt * 100.)
        
            # metrics
            train_measurements_trajectory_over_batch = {
                k:np.array([v]) for k,v in measurements_trajectory.items()
            }
            test_measurements_after_adapt_over_batch = {
                'loss': np.array([test_loss_after_adapt.item()]) , 
                'accu': np.array([test_accu_after_adapt]),
                'aux_loss': np.array([aux_loss])
            }
            update_sum_measurements(sum_test_measurements_after_adapt_over_meta_set,
                                    test_measurements_after_adapt_over_batch)
            update_sum_measurements_trajectory(sum_train_measurements_trajectory_over_meta_set,
                                               train_measurements_trajectory_over_batch)
            n_task_batches += 1

            # logging
            if analysis:
                self.log_output(i,
                    train_measurements_trajectory_over_batch,
                    test_measurements_after_adapt_over_batch,
                    write_tensorboard=False)


        results = {
            'train_loss_trajectory': divide_measurements(
                sum_train_measurements_trajectory_over_meta_set, n=n_task_batches),
            'test_loss_after': divide_measurements(
                sum_test_measurements_after_adapt_over_meta_set, n=n_task_batches)
        }
        mean, i95 = (np.mean(val_task_acc), 
            1.96 * np.std(val_task_acc) / np.sqrt(len(val_task_acc)))
        results['val_task_acc'] = "{:.2f} ± {:.2f} %".format(mean, i95) 
    
        return results


    def get_labels(self, y_batch, n_way, n_shot, n_query, batch_sz):
        # original y_batch: (batch_sz*n_way, n_shot+n_query)
        y_batch = y_batch.reshape(batch_sz, n_way, -1)
        # batch_sz, n_way, n_shot+n_query
        
        for i in range(y_batch.shape[0]):
            uniq_classes = np.unique(y_batch[i, :, :])
            conversion_dict = {v:k for k, v in enumerate(uniq_classes)}
            # convert labels
            for uniq_class in uniq_classes: 
                y_batch[i, y_batch[i]==uniq_class] = conversion_dict[uniq_class]
            
        shots_y = y_batch[:, :, :n_shot]
        query_y = y_batch[:, :, n_shot:]
        shots_y = shots_y.reshape(batch_sz, -1)
        query_y = query_y.reshape(batch_sz, -1)
        return shots_y, query_y
        

    def log_output(self, iteration,
                train_measurements_trajectory_over_batch,
                test_measurements_after_adapt_over_batch,
                write_tensorboard=False, meta_val=False):

        log_array = ['Iteration: {}'.format(iteration)]
        key_list = ['loss', 'accu']
        for key in key_list:
            if not meta_val:
                avg_train_trajectory = np.mean(train_measurements_trajectory_over_batch[key], axis=0)
                avg_test_after = np.mean(test_measurements_after_adapt_over_batch[key])
                avg_train_after = avg_train_trajectory[-1]
            else:
                avg_train_trajectory = train_measurements_trajectory_over_batch[key]
                avg_test_after = test_measurements_after_adapt_over_batch[key]
                avg_train_after = avg_train_trajectory[-1]

            if 'accu' in key:
                log_array.append('train {} after: \t{:.2f}%'.format(key, 100 * avg_train_after))
                log_array.append('test {} after: \t{:.2f}%'.format(key, 100 * avg_test_after))
            else:
                log_array.append('train {} after: \t{:.3f}'.format(key, avg_train_after))
                log_array.append('test {} after: \t{:.3f}'.format(key, avg_test_after))

            if write_tensorboard:
                if meta_val:
                    self._writer.add_scalar('meta_val/train_{}_post'.format(key),
                                                avg_train_trajectory[-1],
                                                iteration)
                    self._writer.add_scalar('meta_val/test_{}_post'.format(key), avg_test_after, iteration)
                else:
                    self._writer.add_scalar('meta_train/train_{}_post'.format(key),
                                                avg_train_trajectory[-1],
                                                iteration)
                    self._writer.add_scalar('meta_train/test_{}_post'.format(key), avg_test_after, iteration)

            log_array.append(' ') 
        if not meta_val:
            tqdm.write('\n'.join(log_array))
