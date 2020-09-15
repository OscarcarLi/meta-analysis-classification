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
from analysis.objectives import var_reduction




def one_hot(indices, depth):
    """
    Returns a one-hot tensor.
    This is a PyTorch equivalent of Tensorflow's tf.one_hot.
        
    Parameters:
      indices:  a (n_batch, m) Tensor or (m) Tensor.
      depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """
    # print(indices)
    encoded_indices = torch.zeros(indices.size() + torch.Size([depth])).cuda()
    index = indices.view(indices.size()+torch.Size([1]))
    encoded_indices = encoded_indices.scatter_(1,index,1)
    
    return encoded_indices


def smooth_loss(logits, labels, num_classes, eps):

    smoothed_one_hot = one_hot(labels.reshape(-1), num_classes)
    smoothed_one_hot = smoothed_one_hot * (1 - eps) + (1 - smoothed_one_hot) * eps / (num_classes - 1)
    log_prb = F.log_softmax(logits.reshape(-1, num_classes), dim=1)
    loss = -(smoothed_one_hot * log_prb).sum(dim=1)
    # print("loss:", loss)
    loss = loss.mean()
    return loss


def get_labels(y_batch, n_way, n_shot, n_query, batch_sz, mgr_n_query=None, rp=None):
    # original y_batch: (batch_sz*n_way, n_shot+n_query)
    y_batch = y_batch.reshape(batch_sz, n_way, -1)
    # batch_sz, n_way, n_shot+n_query
    
    for i in range(y_batch.shape[0]):
        uniq_classes = np.unique(y_batch[i, :, :].cpu().numpy())
        conversion_dict = {v:k for k, v in enumerate(uniq_classes)}
        # convert labels
        for uniq_class in uniq_classes: 
            y_batch[i, y_batch[i]==uniq_class] = conversion_dict[uniq_class]
        
    shots_y = y_batch[:, :, :n_shot]
    if rp is not None:
        query_y = []
        for c in range(n_way):
            indices = rp[(rp>=(c*mgr_n_query)) & (rp<((c+1)*mgr_n_query))] - (c*mgr_n_query)
            query_y.append(y_batch[:, c, n_shot + indices])
        query_y = torch.cat(query_y, dim=1)
    else:
        query_y = y_batch[:, :, n_shot:]
    shots_y = shots_y.reshape(batch_sz, -1)
    query_y = query_y.reshape(batch_sz, -1)
    return shots_y, query_y


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
To train model on all classes together
"""

class Classical_algorithm_trainer(object):

    def __init__(self, model, optimizer, writer, log_interval, save_folder, grad_clip, num_classes,
        loss, aux_loss=None, gamma=0., update_gap=1, label_offset=0, eps=0.):

        self._model = model
        self._loss = loss
        self._aux_loss = aux_loss
        self._gamma = gamma
        self._optimizer = optimizer
        self._writer = writer
        self._log_interval = log_interval 
        self._save_folder = save_folder
        self._grad_clip = grad_clip
        self._label_offset = label_offset
        self._global_iteration = 0
        self._update_gap = update_gap
        self._eps = eps
        self._num_classes = num_classes


    def run(self, dataset_loaders, dataset_mgrs, epoch=None, is_training=True, grad_analysis=False):

        if is_training:
            self._model.train()
        else:
            self._model.eval()

        # statistics init.
        n_param = 0
        for name, param in self._model.named_parameters():
            if param.requires_grad:
                n_param += len(param.flatten()) 
        if grad_analysis:
            first_moment = np.zeros(n_param)
            second_raw_moment = np.zeros(n_param)

        # losses
        erm_loss_name, erm_loss_func = self._loss
        if self._aux_loss is not None:
            aux_loss_name, aux_loss_func = self._aux_loss
            self._gamma = min(0.5, (1.01 * self._gamma))
            print("gamma: {:.4f}".format(self._gamma))

        # loaders and iterators
        iterator = tqdm(enumerate(dataset_loaders[0], start=1),
                        leave=False, file=sys.stdout, position=0)
        tf_iterator = iter(dataset_loaders[1])
        aux_iterator = iter(dataset_loaders[2])


        # metrics aggregation
        aggregate = defaultdict(list)
        val_aggregate = defaultdict(list)
        # torch.autograd.set_detect_anomaly(True)

        # constants
        n_way = dataset_mgrs[0].n_way
        n_shot = dataset_mgrs[0].n_shot
        n_query = dataset_mgrs[0].n_query
        mt_batch_sz = dataset_mgrs[0].batch_size        
        print(f"n_way: {n_way}, n_shot: {n_shot}, n_query: {n_query}, mt_batch_sz: {mt_batch_sz}")

        for i, batch in iterator:
            
            # global iterator count
            self._global_iteration += 1
            
            analysis = (i % self._log_interval == 0)
            batch_size = len(batch)
            
            # fetch samples
            batch_x, batch_y = batch
            batch_y = batch_y - self._label_offset
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            
            # fix BN stats issue with multi-gpu support -> swap ways and shot
            assert batch_x.dim() == 5 # bsz*nway x nshot+n_query x 3 x 84 x 84
            assert batch_y.dim() == 2 # bsz*nway x nshot+n_query
            batch_x = batch_x.reshape(mt_batch_sz, n_way, n_shot+n_query, *(batch_x.shape[-3:]))
            batch_y = batch_y.reshape(mt_batch_sz, n_way, n_shot+n_query)
            batch_x = batch_x.transpose(1, 2)
            # bsz x nshot+n_query x nway x 3 x 84 x 84
            batch_y = batch_y.transpose(1, 2)
            # bsz x nshot+n_query x nway x 3 x 84 x 84                             
            batch_x = batch_x.reshape(-1, *(batch_x.shape[-3:]))
            batch_y = batch_y.reshape(-1)
                                    
            # loss computation + metrics
            if self._model.module.classifier_type == 'cvx-classifier': 
                if self._model.module.fc.lambd < 1.:
                    # aux_batch_x, aux_batch_y = next(aux_iterator)
                    # aux_batch_y = aux_batch_y - self._label_offset
                    # aux_batch_x = aux_batch_x.cuda()
                    # aux_batch_y = aux_batch_y.cuda()
                    # aux_batch_x = aux_batch_x.reshape(mt_batch_sz, n_way, n_shot, *(aux_batch_x.shape[-3:]))
                    # aux_batch_y = aux_batch_y.reshape(mt_batch_sz, n_way, n_shot)
                    # aux_batch_x = aux_batch_x.transpose(1, 2)
                    # aux_batch_y = aux_batch_y.transpose(1, 2)
                    # aux_batch_x = aux_batch_x.reshape(-1, *(aux_batch_x.shape[-3:]))
                    # aux_batch_y = aux_batch_y.reshape(-1)
                    # support_features_x = self._model(torch.cat([batch_x[:n_shot*n_way], aux_batch_x]), features_only=True)
                    # self._model.module.fc.update_L(support_features_x[:n_shot*n_way], batch_y[:n_shot*n_way], 
                    #     support_features_x[n_shot*n_way:], aux_batch_y)
                    support_features_x = self._model(batch_x[:n_shot*n_way], features_only=True)
                    self._model.module.fc.update_L(support_features_x[:n_shot*n_way], batch_y[:n_shot*n_way])
                else:
                    self._model.module.fc.update_L(None, None)

            try:
                query_x, query_y = next(tf_iterator)
            except StopIteration:
                tf_iterator = iter(dataset_loaders[1])
                query_x, query_y = next(tf_iterator)
            query_y = query_y - self._label_offset
            query_x = query_x.cuda()
            query_y = query_y.cuda()
            query_features_x = self._model(query_x, features_only=True)
            output_x = self._model.module.fc(query_features_x)
            erm_loss_value = smooth_loss(
                output_x, query_y, self._num_classes, self._eps)
            loss = erm_loss_value
            aggregate[erm_loss_name].append(erm_loss_value.item())
            accu = None
            if 'cross_ent' == erm_loss_name:
                accu = accuracy(output_x, query_y) * 100.
                aggregate['accu'].append(accu)
                
            if is_training:
                self._optimizer.zero_grad()
                loss.backward()
                if self._grad_clip > 0.:
                    clip_grad_norm_(self._model.parameters(), self._grad_clip)
                if not grad_analysis:
                    self._optimizer.step()
                else:
                    curr = 0
                    for name, param in self._model.named_parameters():
                        if param.requires_grad and 'fc.' not in name:
                            param_len = len(param.flatten())
                            grad = param.grad.flatten().cpu().numpy()
                            assert grad.shape == param.flatten().shape
                            first_moment[curr:curr+param_len] += grad
                            second_raw_moment[curr:curr+param_len] += grad**2
                            curr += param_len

                # var_g = 0.
                # sraw_mom = 0.
                # mean_g_sqr = 0.
                # for group in self._optimizer.param_groups:
                #     for p in group['params']:
                #         if p.grad is None:
                #             continue
                #         state = self._optimizer.state[p]
                #         exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                #         sraw_mom += torch.sum(exp_avg_sq).item()
                #         var_g += torch.sum(exp_avg_sq - exp_avg ** 2).item()
                #         mean_g_sqr += torch.sum(exp_avg ** 2).item()
                # # print(f"variance : {var_g}")
                # aggregate['grad_var_sum'] = var_g 
                # # print(f"sec. raw mom : {sraw_mom}")
                # aggregate['grad_srm_sum'] = sraw_mom 
                # # print(f"mean : {mean}")
                # aggregate['mean_grad_norm'] = np.sqrt(mean_g_sqr) 
                
                # opt_lambd.step()

            # if self._model.module.classifier_type == 'avg-classifier':
            #     self._model.module.fc.update_Lg(
            #         torch.cat([features_x, aux_features_x], dim=0),
            #         torch.cat([batch_y, aux_batch_y], dim=0))
                # self._model.module.fc.project_Lg()
                # self._model.module.fc.update_Lg()

            # logging
            if analysis and is_training and not grad_analysis:
                metrics = {}
                for name, values in aggregate.items():
                    metrics[name] = np.mean(values)
                    val_aggregate["val_" + name].append(np.mean(values))
                self.log_output(epoch, i, metrics)
                aggregate = defaultdict(list)   

        # if self._model.module.classifier_type == 'avg-classifier':
        #     self._model.eval()
        #     print("Updating feature means")
        #     # assert features_x.shape == (len(batch_y), self._model.module.final_feat_dim)
        #     self._model.module.fc.L.weight.fill_(0.)
        #     erm_iterator = tqdm(enumerate(erm_loader, start=1),
        #                 leave=False, file=sys.stdout, position=0)
        #     for i, batch in erm_iterator:
        #         batch_x, batch_y = batch
        #         batch_y = batch_y - self._label_offset
        #         batch_x = batch_x.cuda()
        #         batch_y = batch_y.cuda()
        #         with torch.no_grad():
        #             features_x = self._model(batch_x, features_only=True)
        #             self._model.module.fc.update(features_x, batch_y)
            

        # save model and log tboard for eval
        if is_training and self._save_folder is not None and not grad_analysis:
            save_name = "classical_{0}_{1:03d}.pt".format('resnet', epoch)
            save_path = os.path.join(self._save_folder, save_name)
            with open(save_path, 'wb') as f:
                torch.save({
                    'model': self._model.state_dict(),
                    'optimizer': self._optimizer}, f)

        if not grad_analysis:
            metrics = {}
            for name, values in val_aggregate.items():
                metrics[name] = np.mean(values)
            self.log_output(epoch, None, metrics)    

        else:
            second_raw_moment /= len(erm_loader)
            first_moment /= len(erm_loader)
            self.log_output(epoch, i, {
                "srm_g": np.sum(second_raw_moment),
                "mean_g_norm": np.linalg.norm(first_moment),
                "var_g": np.sum(second_raw_moment - (first_moment ** 2)),
                "uncertainity": np.mean(np.abs(first_moment) / (np.sqrt(second_raw_moment) + 1e-6))
            })



    def log_output(self, epoch, iteration,
                metrics_dict):
        if iteration is not None:
            log_array = ['Epoch {} Iteration {}'.format(epoch, iteration)]
        else:
            log_array = ['Epoch {} '.format(epoch)]
        for key in metrics_dict:
            log_array.append(
                '{}: \t{:.3f}'.format(key, metrics_dict[key]))
            if self._writer is not None:
                self._writer.add_scalar(
                    key, metrics_dict[key], self._global_iteration)
        log_array.append(' ') 
        tqdm.write('\n'.join(log_array))





"""
To train model on all classes together
"""

class TF_algorithm_trainer(object):

    def __init__(self, model, optimizer, writer, log_interval, save_folder, grad_clip, num_classes,
        loss, aux_loss=None, gamma=0., update_gap=1, label_offset=0, eps=0.):

        self._model = model
        self._loss = loss
        self._aux_loss = aux_loss
        self._gamma = gamma
        self._optimizer = optimizer
        self._writer = writer
        self._log_interval = log_interval 
        self._save_folder = save_folder
        self._grad_clip = grad_clip
        self._label_offset = label_offset
        self._global_iteration = 0
        self._update_gap = update_gap
        self._eps = eps
        self._num_classes = num_classes


    def run(self, dataset_loader, epoch=None, is_training=True, grad_analysis=False):

        if is_training:
            self._model.train()
        else:
            self._model.eval()

        # statistics init.
        n_param = 0
        for name, param in self._model.named_parameters():
            if param.requires_grad:
                n_param += len(param.flatten()) 
        if grad_analysis:
            first_moment = np.zeros(n_param)
            second_raw_moment = np.zeros(n_param)

        # losses
        erm_loss_name, erm_loss_func = self._loss
        if self._aux_loss is not None:
            aux_loss_name, aux_loss_func = self._aux_loss
            self._gamma = min(0.5, (1.01 * self._gamma))
            print("gamma: {:.4f}".format(self._gamma))

        # loaders and iterators
        iterator = tqdm(enumerate(dataset_loader, start=1),
                        leave=False, file=sys.stdout, position=0)

        # metrics aggregation
        aggregate = defaultdict(list)
        val_aggregate = defaultdict(list)
        # torch.autograd.set_detect_anomaly(True)

        
        for i, batch in iterator:
            
            # global iterator count
            self._global_iteration += 1
            
            analysis = (i % self._log_interval == 0)
            batch_size = len(batch)
            
            # fetch samples
            batch_x, batch_y = batch
            batch_y = batch_y - self._label_offset
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
                                
            # loss computation + metrics
            if self._model.module.classifier_type == 'cvx-classifier': 
                self._model.module.fc.update_L(None, None)


            query_features_x = self._model(batch_x, features_only=True)
            output_x = self._model.module.fc(query_features_x)
            erm_loss_value = smooth_loss(
                output_x, batch_y, self._num_classes, self._eps)
            loss = erm_loss_value
            aggregate[erm_loss_name].append(erm_loss_value.item())
            accu = None
            if 'cross_ent' == erm_loss_name:
                accu = accuracy(output_x, batch_y) * 100.
                aggregate['accu'].append(accu)
                
            if is_training:
                self._optimizer.zero_grad()
                loss.backward()
                if self._grad_clip > 0.:
                    clip_grad_norm_(self._model.parameters(), self._grad_clip)
                if not grad_analysis:
                    self._optimizer.step()
                else:
                    curr = 0
                    for name, param in self._model.named_parameters():
                        if param.requires_grad and 'fc.' not in name:
                            param_len = len(param.flatten())
                            grad = param.grad.flatten().cpu().numpy()
                            assert grad.shape == param.flatten().shape
                            first_moment[curr:curr+param_len] += grad
                            second_raw_moment[curr:curr+param_len] += grad**2
                            curr += param_len


            # logging
            if analysis and is_training and not grad_analysis:
                metrics = {}
                for name, values in aggregate.items():
                    metrics[name] = np.mean(values)
                    val_aggregate["val_" + name].append(np.mean(values))
                self.log_output(epoch, i, metrics)
                aggregate = defaultdict(list)   
            

        # save model and log tboard for eval
        if is_training and self._save_folder is not None and epoch%5==0 and not grad_analysis:
            save_name = "classical_{0}_{1:03d}.pt".format('resnet', epoch)
            save_path = os.path.join(self._save_folder, save_name)
            with open(save_path, 'wb') as f:
                torch.save({
                    'model': self._model.state_dict(),
                    'optimizer': self._optimizer}, f)

        if not grad_analysis:
            metrics = {}
            for name, values in val_aggregate.items():
                metrics[name] = np.mean(values)
            self.log_output(epoch, None, metrics)    

        else:
            second_raw_moment /= len(erm_loader)
            first_moment /= len(erm_loader)
            self.log_output(epoch, i, {
                "srm_g": np.sum(second_raw_moment),
                "mean_g_norm": np.linalg.norm(first_moment),
                "var_g": np.sum(second_raw_moment - (first_moment ** 2)),
                "uncertainity": np.mean(np.abs(first_moment) / (np.sqrt(second_raw_moment) + 1e-6))
            })



    def log_output(self, epoch, iteration,
                metrics_dict):
        if iteration is not None:
            log_array = ['Epoch {} Iteration {}'.format(epoch, iteration)]
        else:
            log_array = ['Epoch {} '.format(epoch)]
        for key in metrics_dict:
            log_array.append(
                '{}: \t{:.3f}'.format(key, metrics_dict[key]))
            if self._writer is not None:
                self._writer.add_scalar(
                    key, metrics_dict[key], self._global_iteration)
        log_array.append(' ') 
        tqdm.write('\n'.join(log_array))






"""Use a generic feature backbone but adapt it using 
some auxiliary adaptation strategy.
"""

class Generic_adaptation_trainer(object):

    def __init__(self, algorithm, aux_objective, outer_loss_func, outer_optimizer,
            writer, log_interval, model_type, grad_clip=0., n_aux_objective_steps=5,
            label_offset=0):

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
        self._label_offset = label_offset


    def compute_aux_obj(self, x, y):
        orig_shots_shape = x.shape
        features_x = self._algorithm._model(
                x.reshape(-1, *orig_shots_shape[2:])).reshape(*orig_shots_shape[:2], -1)
        feature_sz = features_x.shape[-1]
        aux_loss = self._aux_objective(
            features_x.reshape(-1, feature_sz), y.reshape(-1))
        return aux_loss

    def optimize_auxiliary_obj(self, shots_x, shots_y):
        aux_loss_before_adaptation = None
        for _ in range (self._n_aux_objective_steps):
            self._outer_optimizer.zero_grad()
            aux_loss = self.compute_aux_obj(shots_x, shots_y)
            aux_loss.backward()
            if self._grad_clip > 0.:
                clip_grad_norm_(self._algorithm._model.parameters(), self._grad_clip)
            self._outer_optimizer.step()
            if aux_loss_before_adaptation is None:
                aux_loss_before_adaptation = aux_loss.item()
        return aux_loss_before_adaptation, aux_loss.item()
        

    def run(self, dataset_iterator, dataset_manager, meta_val=False):

        val_task_acc = []
        self._algorithm._model.eval()

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

        if isinstance(self._algorithm._model, torch.nn.DataParallel):
            obj = self._algorithm._model.module
        else:
            obj = self._algorithm._model
        if hasattr(obj, 'fc') and hasattr(obj.fc, 'scale_factor'):
            scale = obj.fc.scale_factor
        else:
            scale = 10.
    
        print("Setting scale to ", scale)


        # iterator
        iterator = tqdm(enumerate(dataset_iterator, start=1),
                        leave=False, file=sys.stdout, initial=1, position=0)
        
        for i, batch in iterator:
        
            ############## covariates #############
            x_batch, y_batch = batch
            y_batch = y_batch - self._label_offset
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
            shots_y, query_y = get_labels(y_batch, n_way=n_way, 
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
            aux_loss_before_adaptation = aux_loss_after_adaptation = 0.
            if self._aux_objective is not None:
                original_state_dict = self._algorithm._model.state_dict()
                self._algorithm._model.train()
                aux_loss_before_adaptation, aux_loss_after_adaptation = self.optimize_auxiliary_obj(
                    shots_x, shots_y)
            self._algorithm._model.eval()

            
            # forward pass on updated model
            logits, measurements_trajectory = self._algorithm.inner_loop_adapt(
                query=query_x, support=shots_x, 
                support_labels=shots_y, scale=scale)
            assert len(set(shots_y)) == len(set(query_y))
            

            # scale = self._algorithm._model.module.fc.scale_factor
            # reshape logits
            logits = scale * logits.reshape(-1, logits.size(-1))
            query_y = query_y.reshape(-1)
            assert logits.size(0) == query_y.size(0)
            analysis = (i % self._log_interval == 0)

            # reinstate original model for the next task
            if self._aux_objective is not None:
                self._algorithm._model.load_state_dict(original_state_dict)


            test_loss_after_adapt = self._outer_loss_func(logits, query_y)
            test_accu_after_adapt = accuracy(logits, query_y) * 100.
            if self._aux_objective is not None:
                test_aux_loss = self.compute_aux_obj(query_x, query_y).item()
            val_task_acc.append(test_accu_after_adapt)
        
            # metrics
            if self._aux_objective is not None:
                measurements_trajectory['aux_loss'] = [aux_loss_after_adaptation]
            train_measurements_trajectory_over_batch = {
                k:np.array([v]) for k,v in measurements_trajectory.items()
            }
            test_measurements_after_adapt_over_batch = {
                'loss': np.array([test_loss_after_adapt.item()]) , 
                'accu': np.array([test_accu_after_adapt])
            }
            if self._aux_objective is not None:
                test_measurements_after_adapt_over_batch['aux_loss'] =  np.array([test_aux_loss])
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
                write_tensorboard=False):

        log_array = ['Iteration: {}'.format(iteration)]
        key_list = ['loss', 'accu', 'aux_loss']
        for key in key_list:
            try:
                avg_train_trajectory = np.mean(train_measurements_trajectory_over_batch[key], axis=0)
                avg_test_after = np.mean(test_measurements_after_adapt_over_batch[key])
                avg_train_after = avg_train_trajectory[-1]
            
                log_array.append('train {} after: \t{:.3f}'.format(key, avg_train_after))
                log_array.append('test {} after: \t{:.3f}'.format(key, avg_test_after))

                if write_tensorboard:
                    self._writer.add_scalar('train_{}_post'.format(key),
                                                avg_train_trajectory[-1],
                                                iteration)
                    self._writer.add_scalar('test_{}_post'.format(key), avg_test_after, iteration)
                log_array.append(' ')
            except:
                continue 
        tqdm.write('\n'.join(log_array))





"""
To meta-learn and transfer-learn the feat. extractor
"""

class MetaClassical_algorithm_trainer(object):

    def __init__(self, model, algorithm, optimizer, writer, log_interval, 
        save_folder, grad_clip, loss_func, label_offset=0):

        self._model = model
        self._algorithm = algorithm
        self._loss_func = loss_func
        self._optimizer = optimizer
        self._writer = writer
        self._log_interval = log_interval 
        self._save_folder = save_folder
        self._grad_clip = grad_clip
        self._label_offset = label_offset
        self._global_iteration = 0
        

    def run(self, mt_loader, mt_manager, n_query, epoch=None, is_training=True):

        if is_training:
            self._model.train()
        else:
            self._model.eval()

        
        # loaders and iterators
        mt_iterator = tqdm(enumerate(mt_loader, start=1),
                        leave=False, file=sys.stdout, position=0)
        
        # metrics aggregation
        aggregate = defaultdict(list)
        
        # constants
        n_way = mt_manager.n_way
        n_shot = mt_manager.n_shot
        mgr_n_query = mt_manager.n_query
        mt_batch_sz = mt_manager.batch_size
        
        print(f"n_way: {n_way}, n_shot: {n_shot}, manager n_query: {mgr_n_query}, actual n_query: {n_query} mt_batch_sz: {mt_batch_sz}")

        # meta-learning loss computation and metrics
        if isinstance(self._algorithm._model, torch.nn.DataParallel):
            obj = self._algorithm._model.module
        else:
            obj = self._algorithm._model
        if hasattr(obj, 'fc') and hasattr(obj.fc, 'scale_factor'):
            scale = obj.fc.scale_factor
            # print("setting scale factor to: ", scale.item())
        elif hasattr(obj, 'scale_factor'):
            scale = obj.scale_factor
        else:
            scale = 1.
        
        print("scale", scale)

        for i, mt_batch in mt_iterator:
            
            # global iterator count
            self._global_iteration += 1
            analysis = (i % self._log_interval == 0)

            # randperm
            if mgr_n_query > n_query:
                rp = np.random.permutation(mgr_n_query * n_way)[:n_query * n_way]
            else:
                rp = None 

            # meta-learning data creation
            mt_batch_x, mt_batch_y = mt_batch
            # mt_batch_x = mt_batch_x.cuda()
            # mt_batch_y = mt_batch_y.cuda()
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
                    indices = rp[(rp>=(c*mgr_n_query)) & (rp<((c+1)*mgr_n_query))] - (c*mgr_n_query)
                    query_x.append(mt_batch_x[:, c, n_shot + indices, :, :, :])
                query_x = torch.cat(query_x, dim=1)
            # (batch_sz, n_way, n_query, channels , height , width)
            shots_x = shots_x.reshape(mt_batch_sz, -1, *original_shape[-3:])
            # (batch_sz, n_way*n_shot, channels , height , width)
            query_x = query_x.reshape(mt_batch_sz, -1, *original_shape[-3:])
            # (batch_sz, n_way*n_query, channels , height , width)
            shots_y, query_y = get_labels(mt_batch_y, n_way=n_way, 
                n_shot=n_shot, n_query=n_query, batch_sz=mt_batch_sz, mgr_n_query=mgr_n_query, rp=rp)
            assert shots_x.shape == (mt_batch_sz, n_way*n_shot, *original_shape[-3:])
            assert query_x.shape == (mt_batch_sz, n_way*n_query, *original_shape[-3:])
            assert shots_y.shape == (mt_batch_sz, n_way*n_shot)
            assert query_y.shape == (mt_batch_sz, n_way*n_query)

            # to cuda
            shots_x = shots_x.cuda()
            query_x = query_x.cuda()
            shots_y = shots_y.cuda()
            query_y = query_y.cuda()
            

            logits, measurements_trajectory = self._algorithm.inner_loop_adapt(
                query=query_x, support=shots_x, 
                support_labels=shots_y, scale=scale)
            assert all(np.unique(shots_y.cpu().numpy()) == np.unique(query_y.cpu().numpy()))
            logits = scale * logits.reshape(-1, logits.size(-1))
            query_y = query_y.reshape(-1)
            assert logits.size(0) == query_y.size(0)
            mt_loss = self._loss_func(logits, query_y)
            mt_accu = accuracy(logits, query_y) * 100.
            aggregate['mt_outer_loss'].append(mt_loss.item())
            aggregate['mt_outer_accu'].append(mt_accu)
            for k in measurements_trajectory:
                aggregate[k].append(measurements_trajectory[k][-1])
            loss = mt_loss

            
            # optimizer step
            if is_training:
                self._optimizer.zero_grad()
                loss.backward()
                if self._grad_clip > 0.:
                    clip_grad_norm_(self._model.parameters(), self._grad_clip)
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
            save_name = "classical_{0}_{1:03d}.pt".format('resnet', epoch)
            save_path = os.path.join(self._save_folder, save_name)
            with open(save_path, 'wb') as f:
                torch.save({'model': self._model.state_dict(),
                           'optimizer': self._optimizer}, f)



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