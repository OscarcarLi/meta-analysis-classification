import os
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import torch
import json
from torch.nn.utils.clip_grad import clip_grad_norm_

from maml.grad import quantile_marks, get_grad_norm_from_parameters
from maml.models.lstm_embedding_model import LSTMAttentionEmbeddingModel
from maml.utils import accuracy
from maml.algorithm import RegMAML_inner_algorithm

class Gradient_based_algorithm_trainer(object):

    def __init__(self, algorithm, outer_loss_func, outer_optimizer, device,
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
        self._device = device 
    
    def run(self, dataset_iterator, is_training=False, start=1, stop=1):
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
            sum_test_loss_after_adapt = 0.0
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
                sum_test_loss_after_adapt += test_loss_after_adapt

                test_measurements_after_adapt_over_batch['loss'].append(test_loss_after_adapt.item())
                if self._algorithm.is_classification:
                    test_measurements_after_adapt_over_batch['accu'].append(
                        accuracy(test_pred_after_adapt, test_task.y)
                    )

            update_sum_measurements_trajectory(sum_train_measurements_trajectory_over_meta_set,
                                               train_measurements_trajectory_over_batch)
            update_sum_measurements(sum_test_measurements_before_adapt_over_meta_set,
                                    test_measurements_before_adapt_over_batch)
            update_sum_measurements(sum_test_measurements_after_adapt_over_meta_set,
                                    test_measurements_after_adapt_over_batch)
            n_tasks += batch_size

            if is_training:
                avg_test_loss_after_adapt = sum_test_loss_after_adapt / batch_size
                # torch.mean(torch.stack(test_measurements_after_adapt_over_batch['loss'])) # make list a torch.tensor
                self._outer_optimizer.zero_grad()
                avg_test_loss_after_adapt.backward() # here back prop will propagate all the way to the initialization parameters
                outer_grad_norm_before_clip = get_grad_norm_from_parameters(self._algorithm._model.parameters())
                self._writer.add_scalar('outer_grad/model_norm/before_clip', outer_grad_norm_before_clip, i)
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
        
        # if not is_training:
        #     self.log_output(
        #         start,
        #         results['train_loss_trajectory'],
        #         results['test_loss_before'],
        #         results['test_loss_after'],
        #         write_tensorboard=True, meta_val=True)

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
                        self._writer.add_scalar('train_{}/after {} step'.format(key, step),
                                                avg_train_trajectory[step],
                                                iteration)
                    self._writer.add_scalar('test_{}/before_update'.format(key), avg_test_before, iteration)
                    self._writer.add_scalar('test_{}/after_update'.format(key), avg_test_after, iteration)

            # std_train_before = np.std(np.array(train_measurements_trajectory_over_batch[key])[:,0])
            # std_train_after = np.std(np.array(train_measurements_trajectory_over_batch[key])[:,-1])
            # std_test_before = np.std(test_measurements_before_adapt_over_batch[key])
            # std_test_after = np.std(test_measurements_after_adapt_over_batch[key])

            # log_array.append('std train {} before: \t{:.3f}'.format(key, std_train_before))
            # log_array.append('std train {} after: \t{:.3f}'.format(key, std_train_after))
            # log_array.append('std test {} before: \t{:.3f}'.format(key, std_test_before))
            # log_array.append('std test {} after: \t{:.3f}'.format(key, std_test_after))
            log_array.append('\n') 
        if not meta_val:
            print('\n'.join(log_array))

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

    def __init__(self, algorithm, outer_loss_func, outer_optimizer, model, embedding_model,
            writer, log_interval, save_interval, model_type, save_folder, outer_loop_grad_norm, device):

        self._algorithm = algorithm
        self._outer_loss_func = outer_loss_func
        self._outer_optimizer = outer_optimizer
        self._writer = writer
        self._log_interval = log_interval # at log_interval will do gradient analysis
        self._save_interval = save_interval
        self._model_type = model_type
        self._save_folder = save_folder
        self._outer_loop_grad_norm = outer_loop_grad_norm
        self._model = model
        self._embedding_model = embedding_model
        self._device = device
        self.to(self._device)
        


    def compute_meta_gradient(self, train_task, test_task, modulation, 
                adapted_params, hessian_inv, outer_params):
        """
        param_dict -> theta
        adapted_params -> phi
        """

        adapted_params = torch.autograd.Variable(adapted_params.detach(),
                            requires_grad=True) 
        # test
        preds_test = self._model(test_task.x, modulation=modulation, 
            update_params=adapted_params)
        loss_test = self._outer_loss_func(preds_test, test_task.y)
        # print(loss_test)
        
        # train
        preds_train = self._model(train_task.x, modulation=modulation, 
            update_params=adapted_params)
        # print (adapted_params)
        loss_train = self._outer_loss_func(preds_train, train_task.y)
        if self._algorithm._l2_lambda > 0.:
            l2_loss = self._algorithm._l2_lambda * adapted_params.pow(2).sum()
            loss_train += l2_loss
        # print(loss_train)
        
        # meta gradient computation
        grad_phi_test = torch.autograd.grad(loss_test, adapted_params,
                    retain_graph=True, create_graph=False, allow_unused=False)[0]
        grad_phi_test = grad_phi_test.flatten()
        
        grad_phi_train = torch.autograd.grad(loss_train, adapted_params,
                    retain_graph=True, create_graph=True, allow_unused=False)[0]
        grad_phi_train = grad_phi_train.flatten()
        
        grad_theta_test_1 =  torch.autograd.grad(loss_test, outer_params,
                    retain_graph=True, create_graph=False, allow_unused=False)

        grad_theta_test_2 = torch.autograd.grad(grad_phi_train.unsqueeze(1), outer_params,
            torch.mm(hessian_inv.t(), grad_phi_test.unsqueeze(1)), create_graph=True)

        meta_grad_list = []
        for grad_1, grad_2 in zip(grad_theta_test_1, grad_theta_test_2):
            meta_grad_list.append(grad_1 - grad_2)

        return meta_grad_list



    def compute_meta_grad(self, batch_train_task, batch_test_task,
        batch_adapted_params, batch_hessian_inv):

        assert len(batch_train_task) == len(batch_test_task) == len(batch_adapted_params)\
                == len(batch_hessian_inv)

        batch_sz = len(batch_train_task)
        outer_params = []
        for param_group in self._outer_optimizer.param_groups:
            outer_params += param_group['params']
        
        for train_task, test_task, adapted_params, hessian_inv in zip(
                batch_train_task, batch_test_task, batch_adapted_params, batch_hessian_inv):
            
            modulation = self._embedding_model(train_task, 
                return_task_embedding=False)
            meta_grads = self.compute_meta_gradient(train_task, test_task,
                modulation, adapted_params, hessian_inv, outer_params)

            for param, meta_grad in zip(outer_params, meta_grads):
                if param.grad is None:
                    param.grad = (meta_grad / batch_sz)
                else:
                    param.grad += (meta_grad / batch_sz)    


    
    def run(self, dataset_iterator, is_training=False, start=1, stop=1):
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
            modulation_analysis = hasattr(self, '_embedding_model') and \
                                       isinstance(self._embedding_model,
                                                  LSTMAttentionEmbeddingModel)

            if analysis and is_training:
                grad_norm_by_step_over_batch = []
                grad_quantiles_by_step_over_batch = defaultdict(list)
                if modulation_analysis:
                    task_modulation_params_over_batch = []

            batch_size = len(train_task_batch)
            sum_test_loss_after_adapt = 0.0
            if is_training:
                batch_adapted_params = []
                batch_hessian_inv = []

            for train_task, test_task in zip(train_task_batch, test_task_batch):

                # get modulation
                modulation = self._embedding_model(train_task, return_task_embedding=False)

                # evalute test loss before adapt over train
                with torch.no_grad():
                    test_pred_before_adapt = self._model(batch=test_task.x, modulation=modulation)
                    test_loss_before_adapt = self._outer_loss_func(test_pred_before_adapt, test_task.y)
                    test_measurements_before_adapt_over_batch['loss'].append(test_loss_before_adapt.item())
                    if self._algorithm.is_classification:
                        test_measurements_before_adapt_over_batch['accu'].append(
                            accuracy(test_pred_before_adapt, test_task.y))

                # adapt according train_task
                adapted_params, hessian_inv, train_measurements_trajectory, info_dict = \
                        self._algorithm.inner_loop_adapt(train_task, self._model, modulation, analysis=analysis and is_training, iter=i)
                if is_training:
                    batch_adapted_params.append(adapted_params)
                    batch_hessian_inv.append(hessian_inv)
                
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
            
                with torch.no_grad():
                    test_pred_after_adapt = self._model(batch=test_task.x, modulation=modulation,
                            update_params=adapted_params)
                    test_loss_after_adapt = self._outer_loss_func(test_pred_after_adapt, test_task.y)
                    sum_test_loss_after_adapt += test_loss_after_adapt.item()
                
                test_measurements_after_adapt_over_batch['loss'].append(test_loss_after_adapt.item())
                if self._algorithm.is_classification:
                    test_measurements_after_adapt_over_batch['accu'].append(
                        accuracy(test_pred_after_adapt, test_task.y)
                    )

            update_sum_measurements_trajectory(sum_train_measurements_trajectory_over_meta_set,
                                               train_measurements_trajectory_over_batch)
            update_sum_measurements(sum_test_measurements_before_adapt_over_meta_set,
                                    test_measurements_before_adapt_over_batch)
            update_sum_measurements(sum_test_measurements_after_adapt_over_meta_set,
                                    test_measurements_after_adapt_over_batch)
            n_tasks += batch_size

            if is_training:
                avg_test_loss_after_adapt = sum_test_loss_after_adapt / batch_size
                # torch.mean(torch.stack(test_measurements_after_adapt_over_batch['loss'])) # make list a torch.tensor
                self._outer_optimizer.zero_grad()
                self.compute_meta_grad(train_task_batch, test_task_batch, 
                    batch_adapted_params, batch_hessian_inv)
                outer_grad_norm_before_clip = get_grad_norm_from_parameters(self._model.parameters())
                self._writer.add_scalar('outer_grad/model_norm/before_clip', outer_grad_norm_before_clip, i)
                if self._outer_loop_grad_norm > 0.:
                    clip_grad_norm_(self._model.parameters(), self._outer_loop_grad_norm)
                    clip_grad_norm_(self._embedding_model.parameters(), self._outer_loop_grad_norm)
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
                    torch.save(self.state_dict(), f)
        
        results = {'train_loss_trajectory': divide_measurements(sum_train_measurements_trajectory_over_meta_set, n=n_tasks),
               'test_loss_before': divide_measurements(sum_test_measurements_before_adapt_over_meta_set, n=n_tasks),
               'test_loss_after': divide_measurements(sum_test_measurements_after_adapt_over_meta_set, n=n_tasks)}
        
        # if not is_training:
        #     self.log_output(
        #         start,
        #         results['train_loss_trajectory'],
        #         results['test_loss_before'],
        #         results['test_loss_after'],
        #         write_tensorboard=True, meta_val=True)

        return results


    def state_dict(self):
        # for model saving and reloading
        return {'model': self._model.state_dict(),
                'embedding_model': self._embedding_model.state_dict()}

    def to(self, device, **kwargs):
        # called in __init__
        self._device = device
        self._model.to(device, **kwargs)
        self._embedding_model.to(device, **kwargs)
        # now for task specific params
        for module in self._model.classifier.values(): 
            module = module.to(device, **kwargs)

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
                        self._writer.add_scalar('train_{}/after {} step'.format(key, step),
                                                avg_train_trajectory[step],
                                                iteration)
                    self._writer.add_scalar('test_{}/before_update'.format(key), avg_test_before, iteration)
                    self._writer.add_scalar('test_{}/after_update'.format(key), avg_test_after, iteration)

            # std_train_before = np.std(np.array(train_measurements_trajectory_over_batch[key])[:,0])
            # std_train_after = np.std(np.array(train_measurements_trajectory_over_batch[key])[:,-1])
            # std_test_before = np.std(test_measurements_before_adapt_over_batch[key])
            # std_test_after = np.std(test_measurements_after_adapt_over_batch[key])

            # log_array.append('std train {} before: \t{:.3f}'.format(key, std_train_before))
            # log_array.append('std train {} after: \t{:.3f}'.format(key, std_train_after))
            # log_array.append('std test {} before: \t{:.3f}'.format(key, std_test_before))
            # log_array.append('std test {} after: \t{:.3f}'.format(key, std_test_after))
            log_array.append('\n') 
        if not meta_val:
            print('\n'.join(log_array))

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