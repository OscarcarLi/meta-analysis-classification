import os
import sys
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_
from maml.datasets.task import Task
import json
import torch.nn as nn

from maml.grad import quantile_marks, get_grad_norm_from_parameters
from maml.models.lstm_embedding_model import LSTMAttentionEmbeddingModel
from maml.utils import accuracy
from maml.algorithm import RegMAML_inner_algorithm

from maml.logistic_regression_utils import logistic_regression_grad_with_respect_to_w
from maml.logistic_regression_utils import logistic_regression_mixed_derivatives_with_respect_to_w_then_to_X

class Gradient_based_algorithm_trainer(object):

    def __init__(self, algorithm, outer_loss_func, outer_optimizer,
            writer, log_interval, save_interval, model_type, save_folder, outer_loop_grad_norm):

        self._algorithm = algorithm
        self._outer_loss_func = outer_loss_func
        self._outer_optimizer = outer_optimizer
        self._writer = writer
        self._log_interval = log_interval # at log_interval will do gradient analysis
        self._save_interval = save_interval
        self._model_type = model_type
        self._save_folder = save_folder
        self._outer_loop_grad_norm = outer_loop_grad_norm

    
    def run(self, dataset_iterator, is_training=False, meta_val=False, start=1, stop=1):
        # looping through the entire meta_dataset once
        sum_train_measurements_trajectory_over_meta_set = defaultdict(float)
        sum_test_measurements_before_adapt_over_meta_set = defaultdict(float)
        sum_test_measurements_after_adapt_over_meta_set = defaultdict(float)
        n_tasks = 0

        for i, (train_task_batch, test_task_batch) in tqdm(enumerate(
                dataset_iterator, start=start if is_training else 1)):
            
            if is_training and i == stop:
                return {'train_loss_trajectory': divide_measurements(sum_train_measurements_trajectory_over_meta_set, n=n_tasks),
                    'test_loss_before': divide_measurements(sum_test_measurements_before_adapt_over_meta_set, n=n_tasks),
                    'test_loss_after': divide_measurements(sum_test_measurements_after_adapt_over_meta_set, n=n_tasks)}

            # _meta_dataset yields data iteration
            train_measurements_trajectory_over_batch = defaultdict(list)
            test_measurements_before_adapt_over_batch = defaultdict(list)
            test_measurements_after_adapt_over_batch = defaultdict(list)
            analysis = (i % self._log_interval == 0)
            modulation_analysis = hasattr(self._algorithm, '_embedding_model') and \
                                       isinstance(self._algorithm._embedding_model,
                                                  LSTMAttentionEmbeddingModel)

            if analysis and is_training:
                grad_norm_by_step_over_batch = []
                grad_quantiles_by_step_over_batch = defaultdict(list)
                if modulation_analysis:
                    task_modulation_params_over_batch = []

            batch_size = len(train_task_batch)

            if is_training:
                self._outer_optimizer.zero_grad()

            # sum_test_loss_after_adapt = 0.0
            for train_task, test_task in zip(train_task_batch, test_task_batch):
                # evalute test loss before adapt over train
                with torch.no_grad():
                    test_pred_before_adapt = self._algorithm.predict_without_adapt(train_task, test_task.x)
                    test_loss_before_adapt = self._outer_loss_func(test_pred_before_adapt, test_task.y)
                    test_measurements_before_adapt_over_batch['loss'].append(test_loss_before_adapt.item())
                    if self._algorithm.is_classification:
                        test_measurements_before_adapt_over_batch['accu'].append(
                            accuracy(test_pred_before_adapt, test_task.y))

                # adapt according train_task
                adapted_param_dict, train_measurements_trajectory, info_dict = \
                        self._algorithm.inner_loop_adapt(train_task, analysis=analysis and is_training, iter=i)
                
                for key, measurements in train_measurements_trajectory.items():
                    train_measurements_trajectory_over_batch[key].append(measurements)

                if analysis and is_training:
                    grad_norm_by_step = info_dict['grad_norm_by_step']
                    grad_quantiles_by_step = info_dict['grad_quantiles_by_step']
                    grad_norm_by_step_over_batch.append(grad_norm_by_step)
                    for step, quantiles in grad_quantiles_by_step.items():
                        grad_quantiles_by_step_over_batch[step].append(quantiles)
                    if modulation_analysis:
                        task_modulation_params = info_dict['layer_modulations']
                        task_modulation_params_over_batch.append(task_modulation_params)
            
                test_pred_after_adapt = self._algorithm.predict_without_adapt(
                        train_task, test_task.x, update_param_dict=adapted_param_dict)
                test_loss_after_adapt = self._outer_loss_func(test_pred_after_adapt, test_task.y)

                test_measurements_after_adapt_over_batch['loss'].append(test_loss_after_adapt.item())
                test_loss_after_adapt /= batch_size # now we are doing this one by one so need to divide individually
                if self._algorithm.is_classification:
                    test_measurements_after_adapt_over_batch['accu'].append(
                        accuracy(test_pred_after_adapt, test_task.y)
                    )
                
                if is_training:
                    test_loss_after_adapt.backward(retain_graph=False, create_graph=False) # here back prop will propagate all the way to the initialization parameters
                

            update_sum_measurements_trajectory(sum_train_measurements_trajectory_over_meta_set,
                                               train_measurements_trajectory_over_batch)
            update_sum_measurements(sum_test_measurements_before_adapt_over_meta_set,
                                    test_measurements_before_adapt_over_batch)
            update_sum_measurements(sum_test_measurements_after_adapt_over_meta_set,
                                    test_measurements_after_adapt_over_batch)
            n_tasks += batch_size

            if is_training:
                outer_grad_norm_before_clip = get_grad_norm_from_parameters(self._algorithm._model.parameters())
                self._writer.add_scalar('outer_grad/model_norm/before_clip', outer_grad_norm_before_clip, i)
                if hasattr(self._algorithm, '_embedding_model'):
                    outer_embedding_model_grad_norm_before_clip = get_grad_norm_from_parameters(self._algorithm._embedding_model.parameters())
                    self._writer.add_scalar('outer_grad/embedding_model_norm/before_clip', outer_embedding_model_grad_norm_before_clip, i)
                if self._outer_loop_grad_norm > 0.:
                    clip_grad_norm_(self._algorithm._model.parameters(), self._outer_loop_grad_norm)
                    #clip_grad_norm_(self._algorithm._embedding_model.parameters(), self._outer_loop_grad_norm)
                self._outer_optimizer.step()

            # logging
            # (i % self._log_interval == 0 or i == 1)
            if analysis and is_training:
                self.log_output(i,
                                train_measurements_trajectory_over_batch,
                                test_measurements_before_adapt_over_batch,
                                test_measurements_after_adapt_over_batch,
                                write_tensorboard=is_training)

                if is_training:
                    self.write_gradient_info_to_board(i,
                                    grad_norm_by_step_over_batch,
                                    grad_quantiles_by_step_over_batch)
                    if modulation_analysis:
                        metadata=[str(t.task_info['task_id']) for t in train_task_batch]
                        self.write_embeddings_output_to_board(task_modulation_params_over_batch, metadata, i)

            # Save model
            if (i % self._save_interval == 0 or i ==1) and is_training:
                save_name = 'maml_{0}_{1}.pt'.format(self._model_type, i)
                save_path = os.path.join(self._save_folder, save_name)
                with open(save_path, 'wb') as f:
                    torch.save(self._algorithm.state_dict(), f)
                
        results = {'train_loss_trajectory': divide_measurements(sum_train_measurements_trajectory_over_meta_set, n=n_tasks),
               'test_loss_before': divide_measurements(sum_test_measurements_before_adapt_over_meta_set, n=n_tasks),
               'test_loss_after': divide_measurements(sum_test_measurements_after_adapt_over_meta_set, n=n_tasks)}
        
        if not is_training and meta_val:
            self.log_output(
                start,
                results['train_loss_trajectory'],
                results['test_loss_before'],
                results['test_loss_after'],
                write_tensorboard=True, meta_val=True)

        return results


    def log_output(self, iteration,
                train_measurements_trajectory_over_batch,
                test_measurements_before_adapt_over_batch,
                test_measurements_after_adapt_over_batch,
                write_tensorboard=False, meta_val=False):

        log_array = ['\nIteration: {}'.format(iteration)]
        key_list = ['loss']
        if self._algorithm.is_classification: key_list.append('accu')
        for key in key_list:
            if not meta_val:
                avg_train_trajectory = np.mean(train_measurements_trajectory_over_batch[key], axis=0)
                avg_test_before = np.mean(test_measurements_before_adapt_over_batch[key])
                avg_test_after = np.mean(test_measurements_after_adapt_over_batch[key])
                avg_train_before = avg_train_trajectory[0]
                avg_train_after = avg_train_trajectory[-1]
            else:
                avg_train_trajectory = train_measurements_trajectory_over_batch[key]
                avg_test_before = test_measurements_before_adapt_over_batch[key]
                avg_test_after = test_measurements_after_adapt_over_batch[key]
                avg_train_before = avg_train_trajectory[0]
                avg_train_after = avg_train_trajectory[-1]

            if key == 'accu':
                log_array.append('train {} before: \t{:.2f}%'.format(key, 100 * avg_train_before))
                log_array.append('train {} after: \t{:.2f}%'.format(key, 100 * avg_train_after))
                log_array.append('test {} before: \t{:.2f}%'.format(key, 100 * avg_test_before))
                log_array.append('test {} after: \t{:.2f}%'.format(key, 100 * avg_test_after))
            else:
                log_array.append('train {} before: \t{:.3f}'.format(key, avg_train_before))
                log_array.append('train {} after: \t{:.3f}'.format(key, avg_train_after))
                log_array.append('test {} before: \t{:.3f}'.format(key, avg_test_before))
                log_array.append('test {} after: \t{:.3f}'.format(key, avg_test_after))

            if write_tensorboard:
                if meta_val:
                    for step in range(0, avg_train_trajectory.shape[0]):
                        self._writer.add_scalar('meta_val_train_{}/after {} step'.format(key, step),
                                                    avg_train_trajectory[step],
                                                    iteration)
                    self._writer.add_scalar('meta_val_test_{}/before_update'.format(key), avg_test_before, iteration)
                    self._writer.add_scalar('meta_val_test_{}/after_update'.format(key), avg_test_after, iteration)
                else:
                    for step in range(0, avg_train_trajectory.shape[0]):
                        self._writer.add_scalar('meta_train_train_{}/after {} step'.format(key, step),
                                                avg_train_trajectory[step],
                                                iteration)
                    self._writer.add_scalar('meta_train_test_{}/before_update'.format(key), avg_test_before, iteration)
                    self._writer.add_scalar('meta_train_test_{}/after_update'.format(key), avg_test_after, iteration)

            # std_train_before = np.std(np.array(train_measurements_trajectory_over_batch[key])[:,0])
            # std_train_after = np.std(np.array(train_measurements_trajectory_over_batch[key])[:,-1])
            # std_test_before = np.std(test_measurements_before_adapt_over_batch[key])
            # std_test_after = np.std(test_measurements_after_adapt_over_batch[key])

            # log_array.append('std train {} before: \t{:.3f}'.format(key, std_train_before))
            # log_array.append('std train {} after: \t{:.3f}'.format(key, std_train_after))
            # log_array.append('std test {} before: \t{:.3f}'.format(key, std_test_before))
            # log_array.append('std test {} after: \t{:.3f}'.format(key, std_test_after))
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
                    'inner_grad/quantile/{}-inner gradient/{} quantile'.format(step_i, qm), quantile_value, iteration)

    
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



class Implicit_Gradient_based_algorithm_trainer(object):

    def __init__(self, algorithm, outer_loss_func, outer_optimizer,
            writer, log_interval, save_interval, model_type, save_folder, outer_loop_grad_norm, hessian_inverse=False):

        self._algorithm = algorithm
        self._outer_loss_func = outer_loss_func
        self._outer_optimizer = outer_optimizer
        self._writer = writer
        self._log_interval = log_interval # at log_interval will do gradient analysis
        self._save_interval = save_interval
        self._model_type = model_type
        self._save_folder = save_folder
        self._outer_loop_grad_norm = outer_loop_grad_norm
        self._hessian_inverse = hessian_inverse


    def run(self, dataset_iterator, is_training=False, meta_val=False, start=1, stop=1):

        if is_training:
            self._algorithm._model.train()
        else:
            self._algorithm._model.eval()

        # looping through the entire meta_dataset once
        sum_train_measurements_trajectory_over_meta_set = defaultdict(float)
        # sum_test_measurements_before_adapt_over_meta_set = defaultdict(float)
        sum_test_measurements_after_adapt_over_meta_set = defaultdict(float)
        n_tasks = 0

        iterator = tqdm(enumerate(dataset_iterator, start=start if is_training else 1),
                        leave=False, file=sys.stdout, initial=start, position=0)
        for i, (train_task_batch, test_task_batch) in iterator:
            if is_training and i == stop:
                return {'train_loss_trajectory': divide_measurements(sum_train_measurements_trajectory_over_meta_set, n=n_tasks),
                    # 'test_loss_before': divide_measurements(sum_test_measurements_before_adapt_over_meta_set, n=n_tasks),
                    'test_loss_after': divide_measurements(sum_test_measurements_after_adapt_over_meta_set, n=n_tasks)}

    
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
                adapted_params, features_train, modulation_train, train_hessian_inv_multiply, train_mixed_partials_left_multiply, train_measurements_trajectory, info_dict = \
                        self._algorithm.inner_loop_adapt(train_task, hessian_inverse=self._hessian_inverse, iter=i) # if hessian_inverse is True then train_hessian is in face train_hessian_inv
                
                for key, measurements in train_measurements_trajectory.items():
                    train_measurements_trajectory_over_batch[key].append(measurements)

                if is_training:
                    features_test = self._algorithm._model(batch=test_task.x, modulation=modulation_train)
                else:
                    with torch.no_grad():
                        features_test = self._algorithm._model(batch=test_task.x, modulation=modulation_train)

                test_pred_after_adapt = self._algorithm._model.scale * F.linear(features_test, weight=adapted_params)
                test_loss_after_adapt = self._outer_loss_func(test_pred_after_adapt, test_task.y)
                
                test_measurements_after_adapt_over_batch['loss'].append(test_loss_after_adapt.item())
                test_loss_after_adapt /= batch_size # now we are doing this one by one so need to divide individually

                if self._algorithm.is_classification:
                    test_measurements_after_adapt_over_batch['accu'].append(
                        accuracy(test_pred_after_adapt, test_task.y)
                    )
                
                if is_training:
                    X_test = features_test.detach().cpu().numpy()
                    y_test = (test_task.y).cpu().numpy()
                    w = adapted_params.detach().cpu().numpy()
                    test_grad_w = logistic_regression_grad_with_respect_to_w(X_test, y_test,
                        self._algorithm._model.scale.detach().cpu().numpy() * w)

                    # train_hessian_inverse = np.linalg.inv(train_hessian)
                    # train_hessian_inv_test_grad = np.matmul(train_hessian_inverse, test_grad_w)
                    # if not self._hessian_inverse:
                    #     train_hessian_inv_test_grad = np.linalg.solve(train_hessian, test_grad_w)
                    # else:
                    #     # in this case train_hessian in actually train_hessian_inv
                    #     train_hessian_inv_args = train_hessian
                    #     train_hessian_inv_test_grad = self._algorithm.compute_inverse_hessian_test_grad(
                    #         train_hessian_inv_args + [test_grad_w])
                    train_hessian_inv_test_grad = train_hessian_inv_multiply(test_grad_w)

                    # test loss's gradient with respect to training set's feature
                    # test_grad_features_train = - np.matmul(train_mixed_partials.T, train_hessian_inv_test_grad)
                    test_grad_features_train = - train_mixed_partials_left_multiply(train_hessian_inv_test_grad)

                    test_grad_features_train = test_grad_features_train.reshape(features_train.shape)

                    features_train.backward(gradient=(torch.tensor(test_grad_features_train,
                                                                      device=self._algorithm._device) / batch_size),
                                           retain_graph=True,
                                           create_graph=False)
                    test_loss_after_adapt.backward(retain_graph=False, create_graph=False)

            update_sum_measurements_trajectory(sum_train_measurements_trajectory_over_meta_set,
                                               train_measurements_trajectory_over_batch)
            # update_sum_measurements(sum_test_measurements_before_adapt_over_meta_set,
            #                         test_measurements_before_adapt_over_batch)
            update_sum_measurements(sum_test_measurements_after_adapt_over_meta_set,
                                    test_measurements_after_adapt_over_batch)
            n_tasks += batch_size

            if is_training:
                outer_model_grad_norm_before_clip = get_grad_norm_from_parameters(self._algorithm._model.parameters())
                self._writer.add_scalar('outer_grad/model_norm/before_clip', outer_model_grad_norm_before_clip, i)
                if self._algorithm._embedding_model:
                    outer_embedding_model_grad_norm_before_clip = get_grad_norm_from_parameters(self._algorithm._embedding_model.parameters())
                    self._writer.add_scalar('outer_grad/embedding_model_norm/before_clip', outer_embedding_model_grad_norm_before_clip, i)
                if self._outer_loop_grad_norm > 0.:
                    clip_grad_norm_(self._algorithm._model.parameters(), self._outer_loop_grad_norm)
                    if self._algorithm._embedding_model:
                        clip_grad_norm_(self._algorithm._embedding_model.parameters(), self._outer_loop_grad_norm)
                self._outer_optimizer.step()

            # logging
            # (i % self._log_interval == 0 or i == 1)
            if analysis and is_training:
                self.log_output(i,
                                train_measurements_trajectory_over_batch,
                                # test_measurements_before_adapt_over_batch,
                                test_measurements_after_adapt_over_batch,
                                write_tensorboard=is_training)


            # Save model
            if (i % self._save_interval == 0 or i ==1) and is_training:
                save_name = 'maml_{0}_{1}.pt'.format(self._model_type, i)
                save_path = os.path.join(self._save_folder, save_name)
                with open(save_path, 'wb') as f:
                    torch.save(self._algorithm.state_dict(), f)
        
        results = {'train_loss_trajectory': divide_measurements(sum_train_measurements_trajectory_over_meta_set, n=n_tasks),
            #    'test_loss_before': divide_measurements(sum_test_measurements_before_adapt_over_meta_set, n=n_tasks),
               'test_loss_after': divide_measurements(sum_test_measurements_after_adapt_over_meta_set, n=n_tasks)}
        
        if (not is_training) and meta_val:
            self.log_output(
                start,
                results['train_loss_trajectory'],
                results['test_loss_after'],
                write_tensorboard=True, meta_val=True)
        
        return results


    def log_output(self, iteration,
                train_measurements_trajectory_over_batch,
                # test_measurements_before_adapt_over_batch,
                test_measurements_after_adapt_over_batch,
                write_tensorboard=False, meta_val=False):

        log_array = ['Iteration: {}'.format(iteration)]
        key_list = ['loss']
        if self._algorithm.is_classification: key_list.append('accu')
        for key in key_list:
            if not meta_val:
                avg_train_trajectory = np.mean(train_measurements_trajectory_over_batch[key], axis=0)
                # avg_test_before = np.mean(test_measurements_before_adapt_over_batch[key])
                avg_test_after = np.mean(test_measurements_after_adapt_over_batch[key])
                # avg_train_before = avg_train_trajectory[0]
                avg_train_after = avg_train_trajectory[-1]
            else:
                avg_train_trajectory = train_measurements_trajectory_over_batch[key]
                # avg_test_before = test_measurements_before_adapt_over_batch[key]
                avg_test_after = test_measurements_after_adapt_over_batch[key]
                # avg_train_before = avg_train_trajectory[0]
                avg_train_after = avg_train_trajectory[-1]

            if key == 'accu':
                # log_array.append('train {} before: \t{:.2f}%'.format(key, 100 * avg_train_before))
                log_array.append('train {} after: \t{:.2f}%'.format(key, 100 * avg_train_after))
                # log_array.append('test {} before: \t{:.2f}%'.format(key, 100 * avg_test_before))
                log_array.append('test {} after: \t{:.2f}%'.format(key, 100 * avg_test_after))
            else:
                # log_array.append('train {} before: \t{:.3f}'.format(key, avg_train_before))
                log_array.append('train {} after: \t{:.3f}'.format(key, avg_train_after))
                # log_array.append('test {} before: \t{:.3f}'.format(key, avg_test_before))
                log_array.append('test {} after: \t{:.3f}'.format(key, avg_test_after))

            if write_tensorboard:
                if meta_val:
                    self._writer.add_scalar('meta_val/train_{}_post'.format(key),
                                                avg_train_trajectory[-1],
                                                iteration)
                    # self._writer.add_scalar('meta_val_test_{}/before_update'.format(key), avg_test_before, iteration)
                    self._writer.add_scalar('meta_val/test_{}_post'.format(key), avg_test_after, iteration)
                else:
                    self._writer.add_scalar('meta_train/train_{}_post'.format(key),
                                                avg_train_trajectory[-1],
                                                iteration)
                    # self._writer.add_scalar('test_{}/before_update'.format(key), avg_test_before, iteration)
                    self._writer.add_scalar('meta_train/test_{}_post'.format(key), avg_test_after, iteration)

            # std_train_before = np.std(np.array(train_measurements_trajectory_over_batch[key])[:,0])
            # std_train_after = np.std(np.array(train_measurements_trajectory_over_batch[key])[:,-1])
            # std_test_before = np.std(test_measurements_before_adapt_over_batch[key])
            # std_test_after = np.std(test_measurements_after_adapt_over_batch[key])

            # log_array.append('std train {} before: \t{:.3f}'.format(key, std_train_before))
            # log_array.append('std train {} after: \t{:.3f}'.format(key, std_train_after))
            # log_array.append('std test {} before: \t{:.3f}'.format(key, std_test_before))
            # log_array.append('std test {} after: \t{:.3f}'.format(key, std_test_after))
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
                    'inner_grad/quantile/{}-inner gradient/{} quantile'.format(step_i, qm), quantile_value, iteration)

    
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

class InnerSolver_algorithm_trainer(object):

    def __init__(self, algorithm, outer_loss_func, outer_optimizer,
            writer, log_interval, save_interval, save_folder, model_type, outer_loop_grad_norm):

        self._algorithm = algorithm
        self._outer_loss_func = outer_loss_func
        self._outer_optimizer = outer_optimizer
        self._writer = writer
        self._log_interval = log_interval # at log_interval will do gradient analysis
        self._save_interval = save_interval
        self._save_folder = save_folder
        self._outer_loop_grad_norm = outer_loop_grad_norm
        self._model_type = model_type 
        

    def run(self, dataset_iterator, is_training=False, meta_val=False, start=1, stop=1):

        if is_training:
            self._algorithm._model.train()
        else:
            self._algorithm._model.eval()

        # looping through the entire meta_dataset once
        sum_train_measurements_trajectory_over_meta_set = defaultdict(float)
        # sum_test_measurements_before_adapt_over_meta_set = defaultdict(float)
        sum_test_measurements_after_adapt_over_meta_set = defaultdict(float)
        n_task_batches = 0

        iterator = tqdm(enumerate(dataset_iterator, start=start if is_training else 1),
                        leave=False, file=sys.stdout, initial=start, position=0)
        
        for i, (train_task_batch, test_task_batch) in iterator:

            if is_training and i == stop:
                return {'train_loss_trajectory': divide_measurements(sum_train_measurements_trajectory_over_meta_set, n=n_task_batches),
                    'test_loss_after': divide_measurements(sum_test_measurements_after_adapt_over_meta_set, n=n_task_batches)}

            batch_size = len(train_task_batch)
            train_task_batch_x = torch.stack([task.x for task in train_task_batch], dim=0)
            # a tensor of size len(train_task_batch) x n_train x n_way
            train_task_batch_y = torch.stack([task.y for task in train_task_batch], dim=0)
            # a tensor of size len(train_task_batch) x n_train

            test_task_batch_x = torch.stack([task.x for task in test_task_batch], dim=0)
            # a tensor of size len(test_task_batch) x n_test x n_way
            test_task_batch_y = torch.stack([task.y for task in test_task_batch], dim=0)
            # a tensor of size len(test_task_batch) x n_test x n_way    

            if is_training:
                self._outer_optimizer.zero_grad()
            
            if is_training:    
                logits, measurements_trajectory = self._algorithm.inner_loop_adapt(
                    query=test_task_batch_x, support=train_task_batch_x, support_labels=train_task_batch_y)
                assert len(set(train_task_batch_y)) == len(set(test_task_batch_y))
                # len(train_task_batch) x n_test x n_way
            else:
                with torch.no_grad():
                    logits, measurements_trajectory = self._algorithm.inner_loop_adapt(
                        query=test_task_batch_x, support=train_task_batch_x, support_labels=train_task_batch_y)
                    assert len(set(train_task_batch_y)) == len(set(test_task_batch_y))
                   
            # reshape logits
            logits = self._algorithm._model.scale * logits.reshape(-1, logits.size(-1))
            test_task_batch_y = test_task_batch_y.reshape(-1)
            assert logits.size(0) == test_task_batch_y.size(0)
            analysis = (i % self._log_interval == 0)

            # compute loss abd accu
            test_loss_after_adapt = self._outer_loss_func(logits, test_task_batch_y)
            test_accu_after_adapt = accuracy(logits, test_task_batch_y)

            if is_training:
                test_loss_after_adapt.backward()
        
            # metrics
            train_measurements_trajectory_over_batch = {
                'loss': np.array([measurements_trajectory['loss']]) , 
                'accu': np.array([measurements_trajectory['accu']])
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

            if is_training:
                outer_model_grad_norm_before_clip = get_grad_norm_from_parameters(self._algorithm._model.parameters())
                self._writer.add_scalar('outer_grad/model_norm/before_clip', outer_model_grad_norm_before_clip, i)
                if self._outer_loop_grad_norm > 0.:
                    clip_grad_norm_(self._algorithm._model.parameters(), self._outer_loop_grad_norm)
                    if self._algorithm._embedding_model:
                        clip_grad_norm_(self._algorithm._embedding_model.parameters(), self._outer_loop_grad_norm)
                self._outer_optimizer.step()

            # logging
            # (i % self._log_interval == 0 or i == 1)
            if analysis and is_training:
                self.log_output(i,
                                train_measurements_trajectory_over_batch,
                                test_measurements_after_adapt_over_batch,
                                write_tensorboard=is_training)


            # Save model
            if (i % self._save_interval == 0 or i ==1) and is_training:
                save_name = 'maml_{0}_{1}.pt'.format(self._model_type, i)
                save_path = os.path.join(self._save_folder, save_name)
                with open(save_path, 'wb') as f:
                    torch.save(self._algorithm.state_dict(), f)
        
        results = {
            'train_loss_trajectory': divide_measurements(sum_train_measurements_trajectory_over_meta_set, n=n_task_batches),
            'test_loss_after': divide_measurements(sum_test_measurements_after_adapt_over_meta_set, n=n_task_batches)
        }
        
        if (not is_training) and meta_val:
            self.log_output(
                start,
                results['train_loss_trajectory'],
                results['test_loss_after'],
                write_tensorboard=True, meta_val=True)
        
        return results


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
                'inner_grad/norm/{}-inner gradient step'.format(step_i), grad_norm, iteration)
        for step_i, quantiles in avg_grad_quantiles_by_step.items():
            for qm, quantile_value in zip(quantile_marks, quantiles):
                self._writer.add_scalar(
                    'inner_grad/quantile/{}-inner gradient/{} quantile'.format(step_i, qm), quantile_value, iteration)

    
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