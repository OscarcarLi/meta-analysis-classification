from collections import defaultdict, OrderedDict
import warnings
import time
import numpy as np
import itertools
from copy import deepcopy
from itertools import chain

# torch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

# ours
from algorithm_trainer.algorithms.grad import soft_clip, get_grad_norm, get_grad_quantiles
from algorithm_trainer.utils import accuracy
from algorithm_trainer.algorithms.logistic_regression_utils import logistic_regression_hessian_pieces_with_respect_to_w, logistic_regression_hessian_with_respect_to_w, logistic_regression_mixed_derivatives_with_respect_to_w_then_to_X, logistic_regression_mixed_derivatives_with_respect_to_w_then_to_X_left_multiply
from algorithm_trainer.utils import spectral_norm
from algorithm_trainer.models.resnet_2 import distLinear, gaussianDA, orthonormalDistLinear

# metaoptnet
from algorithm_trainer.algorithms.metaoptnet_utils import one_hot, computeGramMatrix, binv, batched_kronecker
from qpth.qp import QPFunction

# sklearn
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter("always", ConvergenceWarning)

class Algorithm(object):
    def __init__(self):
        pass

    def inner_loop_adapt(self, task, num_updates=None, analysis=False, iter=None):
        raise NotImplementedError('Subclass should implement inner_loop')

    def predict_without_adapt(self, train_task, batch, param_dict=None):
        raise NotImplementedError()



DEFAULT_MEMO = dict()


def copy_and_replace(original, replace=None, do_not_copy=None):
    """
    A convenience function for creating modified copies out-of-place with deepcopy
    :param original: object to be copied
    :param replace: a dictionary {old object -> new object}, replace all occurences of old object with new object
    :param do_not_copy: a sequence of objects that will not be copied (but may be replaced)
    :return: a copy of obj with replacements
    """
    replace, do_not_copy = replace or {}, do_not_copy or {}
    memo = dict(DEFAULT_MEMO)
    for item in do_not_copy:
        memo[id(item)] = item
    for item, replacement in replace.items():
        memo[id(item)] = replacement
    return deepcopy(original, memo)




class InitBasedAlgorithm(Algorithm):

    def __init__(self, model, loss_func, device, 
            n_way, n_shot, n_query, alpha, num_updates, method, inner_loop_grad_clip):
        
        self._model = model
        self._device = device
        self._loss_func = loss_func
        self._n_way = n_way
        self._n_shot = n_shot
        self._n_query = n_query
        self._alpha = alpha # inner loop lr
        self._num_updates = num_updates
        self._inner_loop_grad_clip = inner_loop_grad_clip
        self._method = method
        self._first_order = (self._method == 'FOMAML')
        print("Init based Algorithm: ", self._method)
        self.to(self._device)

        

    def get_logits(self, model, X):
        # compute loss on support set
        orig_X_shape = X.shape
        logits = model(
            X.reshape(-1, *orig_X_shape[2:]), features_only=False).reshape(*orig_X_shape[:2], -1)        
        return logits


    def compute_gradient_wrt_model(self, X, y, model, create_graph):
        """Compute gradient of self._loss_func(X, y; model),
        based on support, support_labels set but with respect to parameters in model
        """
        
        # compute logits wrt param_dict if param_dict is not None
        logits = self.get_logits(model=model, X=X)
        logits = logits.reshape(-1, logits.size(-1))
        y = y.reshape(-1)
        loss = self._loss_func(logits, y)
        # print("loss", loss)
        # for n, p in self._model.named_parameters():
        #     print(n, torch.norm(p))
        accu = accuracy(logits, y)
        grad_list = torch.autograd.grad(loss, model.parameters(),
                                    create_graph=create_graph, allow_unused=False, only_inputs=True)
        # allow_unused If False, specifying inputs that were not used when computing outputs
        # (and therefore their grad is always zero) is an error. Defaults to False.
        return loss, accu, grad_list


    def get_updated_model(self, model, grad_list):
        """ model_param = model_param - alpha * grad_list
        """
        updates = []
        for (name, param), grad in zip(model.named_parameters(), grad_list):
            # grad will be torch.Tensor
            assert grad is not None, f"Grad is None for {name}"
            if self._inner_loop_grad_clip > 0:
                grad = grad.clamp(min=-self._inner_loop_grad_clip,
                    max=self._inner_loop_grad_clip)
            update = param - self._alpha * grad
            updates.append(update)
        updates = dict(zip(model.parameters(), updates))
        do_not_copy = [tensor for tensor in chain(model.parameters(), model.buffers())
                if tensor not in updates]
        # print("do_not_copy", do_not_copy)
        return copy_and_replace(model, updates, do_not_copy=do_not_copy)
                

    # def get_cloned_params(self):
    #     """ get a dictionary of cloned model params
    #     """
    #     original_params = []
    #     for param in self._model.parameters():
    #         if param.grad is None:
    #             param.grad = torch.zeros_like(param)
    #         original_params.append(param.clone().detach())
    #         original_params[-1].requires_grad_(True)
    #         original_params[-1].grad = param.grad
    #     return original_params


    def get_param_diff(self, params_1, params_2):
        """ returns params_1 - params_2
        """
        diff = []
        for param_1, param_2 in zip(params_1, params_2):
            diff.append(param_1 - param_2)
        return diff


    def populate_grad(self, grad_list):
        """Take values in grad list and populate param.grad with it
        for param in model parameters.
        """
        for param, calculated_grad in zip(self._model.parameters(), grad_list):
            if param.grad is None:
                param.grad = torch.zeros_like(param)  
            param.grad += calculated_grad.detach()
        


    def inner_loop_adapt(self, support, support_labels, query, query_labels):

        
        # adapt means doing the complete inner loop update
        measurements_trajectory = defaultdict(list)
        # copy every tenso's data in the original dictionary
        updated_model = self._model
        # self._model = self.get_cloned_params()
        # maintain model parameters as a dictionary. we will receive gradients wrt to these parameters only
        # adapted_param_dict = OrderedDict(self._model.named_parameters())
         
        
        assert self._num_updates > 0
        for i in range(self._num_updates):
            support_loss, support_accu, grad_list = self.compute_gradient_wrt_model(
                X=support, y=support_labels, model=updated_model,
                create_graph=not self._first_order)
            updated_model = self.get_updated_model(model=updated_model, grad_list=grad_list)
            

        # with torch.no_grad(): # compute the support loss after the last adaptation 
        #     support_logits = self.set_params_and_get_logits(param_dict=adapted_param_dict, X=support) 
        #     support_loss = self._loss_func(support_logits, support_labels)
        #     measurements_trajectory['loss'].append(loss.item())
        #     measurements_trajectory['accu'].append(accuracy(support_logits, support_labels))


        # Now compute loss on query set and from that the outer gradient
        if self._method == 'MAML':
            query_loss, query_accu, outer_grad_list = self.compute_gradient_wrt_model(
                X=query, y=query_labels, model=self._model,
                create_graph=False)
        elif self._method == 'FOMAML':
            query_loss, query_accu, outer_grad_list = self.compute_gradient_wrt_model(
                X=query, y=query_labels, model=updated_model,
                create_graph=False)
        elif self._method == 'Reptile': 
            query_loss, query_accu, grad_list = self.compute_gradient_wrt_model(
                X=query, y=query_labels, model=updated_model,
                create_graph=False)
            updated_model = self.get_updated_model(model=updated_model, grad_list=grad_list)
            outer_grad_list = self.get_param_diff(self._model.parameters(), updated_model.parameters())
        else:
            raise ValueError("Meta-alg not implemented.")
            
        # load back the original parameters
        # original = dict(zip(self._model.parameters(), original_params))
        # do_not_copy = [tensor for tensor in chain(self._model.parameters(), self._model.buffers())
        #         if tensor not in original]
        # self._model = copy_and_replace(self._model, original, do_not_copy=do_not_copy)
        # populate model.grad with outer_grad_list
        self.populate_grad(outer_grad_list)
        
        # metrics
        measurements_trajectory['loss'].append(support_loss.item())
        measurements_trajectory['accu'].append(support_accu * 100.)
        measurements_trajectory['mt_outer_loss'].append(query_loss.item())
        measurements_trajectory['mt_outer_accu'].append(query_accu * 100.)
        # print(measurements_trajectory)
        return measurements_trajectory


    # def get_cloned_param_dict(model):
    #     """ get a dictionary of cloned model params
    #     """
    #     param_dict = OrderedDict(model.named_parameters())
    #     for name, param in param_dict.items():
    #         param_dict[name] = param.clone() 
    #     return param_dict  


    # def update_grad_avg(grad_list_avg, grad_list):
    #     with torch.no_grad():
    #         if grad_list_avg is None:
    #             grad_list_avg = []
    #             for i, grad in enumerate(grad_list):
    #                 grad_list_avg.append(grad / self._num_updates)
    #         else:
    #             for i, grad in enumerate(grad_list):
    #                 assert grad.shape == grad_list_avg[i].shape
    #                 grad_list_avg[i] = grad_list_avg[i] + (grad / self._num_updates)
    #     return grad_list_avg


    def to(self, device, **kwargs):
        self._device = device
        self._model.to(device, **kwargs)


    def state_dict(self):
        # for model saving and reloading
        return {'model': self._model.state_dict()}







class LR(Algorithm):
    def __init__(self, model,
                inner_loss_func, l2_lambda,
                device, is_classification=True):

        self._model = model
        self._inner_loss_func = inner_loss_func
        self._l2_lambda = l2_lambda
        # self._Cs = [1 / lambda_ for lambda_ in self._l2_lambda]
        # print('l2_lambda', self._l2_lambda)
        # print('C_s', self._Cs)
        self._device = device
        self.to(self._device)
        self.is_classification = is_classification

    @staticmethod
    def compute_hessian(X, y, w, l2_lambda):
        lr_hessian = logistic_regression_hessian_with_respect_to_w(X, y, w)
        hessian = lr_hessian + l2_lambda * np.eye(lr_hessian.shape[0])
        return hessian
    
    @staticmethod
    def compute_inverse_hessian_multiply_vector(X, y, w, l2_lambda, v):
        '''
        X  N, (d+1) last dimension the bias
        y  N integer class identity
        w  C, (d+1) number of classes
        v  C(d+1), 1
        '''
        diag, Xbar = logistic_regression_hessian_pieces_with_respect_to_w(X, y, w) # Xbar shape N(C+1), C(d+1)
        # Question: why is this taking much longer than run alone in a terminal
        pre_inv = np.matmul(Xbar, Xbar.T)

        pre_inv = pre_inv + l2_lambda * np.diag(np.reciprocal(diag))
        # np.fill_diagonal(a=pre_inv, val=np.diag(pre_inv) + l2_lambda * np.reciprocal(diag))

        inv = np.linalg.inv(pre_inv)

        # result = 1 / l2_lambda * (v -\
        #      np.matmul(np.matmul(Xbar.T, inv), np.matmul(Xbar, v))) # this was suboptimal
        result = 1 / l2_lambda * (v - np.matmul(Xbar.T, np.matmul(inv, np.matmul(Xbar, v))))

        return result


    def inner_loop_adapt(self, task, hessian_inverse=False, num_updates=None, analysis=False, iter=None, 
            return_estimator=False):
        # adapt means doing the complete inner loop update
        
        measurements_trajectory = defaultdict(list)
        
        # here the features are padded with 1's at the end
        features = self._model(task.x)

        X = features.detach().cpu().numpy()
        y = (task.y).cpu().numpy()

        with warnings.catch_warnings(record=True) as wn:
            lr_model = LogisticRegression(solver='lbfgs', penalty='l2', 
                C=1/(self._l2_lambda), # now use _l2_lambda instead of 2 * _l2_lambda
                tol=1e-6, max_iter=150,
                multi_class='multinomial', fit_intercept=False)
            lr_model.fit(X, y)
        
        # print(lr_model.n_iter_)
        adapted_params = torch.tensor(lr_model.coef_, device=self._device, dtype=torch.float32, requires_grad=False)
        
        if return_estimator:
            return adapted_params

        preds = F.linear(features, weight=adapted_params)

        loss = self._inner_loss_func(preds, task.y)
        measurements_trajectory['loss'].append(loss.item())
        if self.is_classification: 
            measurements_trajectory['accu'].append(accuracy(preds, task.y))

        info_dict = None
        
        l2_lambda_chosen = self._l2_lambda
        # assert np.all(lr_model.C_ == lr_model.C_[0]) # when using multinomial all the chosen C should be the same for each class
        # l2_lambda_chosen = 1 / lr_model.C_[0]
        # print(l2_lambda_chosen)

        # h_inv_multiply, given a vector v of shape C(d+1), 1 returns hessian^-1 @ v
        if not hessian_inverse:
            hessian = self.compute_hessian(X=X, y=y, w=lr_model.coef_, l2_lambda=l2_lambda_chosen)
            h_inv_multiply = lambda v: np.linalg.solve(hessian, v)
        else:
            h_inv_multiply = lambda v: self.compute_inverse_hessian_multiply_vector(X=X, y=y, w=lr_model.coef_.astype(np.float32), l2_lambda=l2_lambda_chosen, v=v)
        
        # mixed_partials_func given a vector v of shape C(d+1), 1
        mixed_partials_left_multiply = lambda v: logistic_regression_mixed_derivatives_with_respect_to_w_then_to_X_left_multiply(X=X, y=y, w=lr_model.coef_.astype(np.float32), a=v)

        return adapted_params, features, h_inv_multiply, mixed_partials_left_multiply, measurements_trajectory, info_dict


    def to(self, device, **kwargs):
        # called in __init__
        self._device = device
        self._model.to(device, **kwargs)
        
    def state_dict(self):
        # for model saving and reloading
        return {'model': self._model.state_dict()}


class SVM(Algorithm):

    def __init__(self, model, inner_loss_func, device, n_way, n_shot, n_query,
        C_reg=0.1, max_iter=15, double_precision=False):
        
        self._model = model
        self._device = device
        self._inner_loss_func = inner_loss_func
        self._C_reg = C_reg
        self._max_iter = max_iter
        self._n_way = n_way
        self._n_shot = n_shot
        self._n_query = n_query
        self._double_precision = double_precision
        self.to(self._device)
   

    def inner_loop_adapt(self, support, support_labels, query=None, return_estimator=False, scale=None):
        """
        Fits the support set with multi-class SVM and 
        returns the classification score on the query set.
        
        This is the multi-class SVM presented in:
        On the Algorithmic Implementation of Multiclass Kernel-based Vector Machines
        (Crammer and Singer, Journal of Machine Learning Research 2001).
        This model is the classification head that we use for the final version.
        Parameters:
        query:  a (tasks_per_batch, n_query, c, h, w) Tensor.
        support:  a (tasks_per_batch, n_support, c, h, w) Tensor.
        support_labels: a (tasks_per_batch, n_support) Tensor.
        n_way: a scalar. Represents the number of classes in a few-shot classification task.
        n_shot: a scalar. Represents the number of support examples given per class.
        C_reg: a scalar. Represents the cost parameter C in SVM.
        Returns: a (tasks_per_batch, n_query, n_way) Tensor.
        """

        measurements_trajectory = defaultdict(list)

        assert(query.dim() == 5)
        assert(support.dim() == 5)
        
        # get features
        orig_query_shape = query.shape
        orig_support_shape = support.shape
        support = self._model(
            support.reshape(-1, *orig_support_shape[2:]), features_only=True).reshape(*orig_support_shape[:2], -1)
        query = self._model(
            query.reshape(-1, *orig_query_shape[2:]), features_only=True).reshape(*orig_query_shape[:2], -1)
                

        tasks_per_batch = query.size(0)
        total_n_support = support.size(1) # support samples across all classes in a task
        total_n_query = query.size(1)     # query samples across all classes in a task
        d = query.size(2)                 # dimension

        n_way = self._n_way               # n_classes in a task
        n_query = self._n_query           # n_query samples per class
        n_shot = self._n_shot             # n_support samples per class
        C_reg = self._C_reg
        maxIter = self._max_iter

        assert(query.dim() == 3)
        assert(support.dim() == 3)
        assert(query.size(0) == support.size(0) and query.size(2) == support.size(2))
        assert(total_n_support == n_way * n_shot)      # total_n_support must equal to n_way * n_shot
        assert(total_n_query == n_way * n_query)      # total_n_query must equal to n_way * n_query


        #Here we solve the dual problem:
        #Note that the classes are indexed by m & samples are indexed by i.
        #min_{\alpha}  0.5 \sum_m ||w_m(\alpha)||^2 + \sum_i \sum_m e^m_i alpha^m_i
        #s.t.  \alpha^m_i <= C^m_i \forall m,i , \sum_m \alpha^m_i=0 \forall i

        #where w_m(\alpha) = \sum_i \alpha^m_i x_i,
        #and C^m_i = C if m  = y_i,
        #C^m_i = 0 if m != y_i.
        #This borrows the notation of liblinear.
        
        #\alpha is an (total_n_support, n_way) matrix
        kernel_matrix = computeGramMatrix(support, support)

        id_matrix_0 = torch.eye(n_way).expand(tasks_per_batch, n_way, n_way).cuda()
        block_kernel_matrix = batched_kronecker(kernel_matrix, id_matrix_0)
        #This seems to help avoid PSD error from the QP solver.
        block_kernel_matrix += 1.0 * torch.eye(n_way*total_n_support).expand(
            tasks_per_batch, n_way*total_n_support, n_way*total_n_support).cuda()
        
        support_labels_one_hot = one_hot(support_labels.view(tasks_per_batch * total_n_support), n_way) 
        # (tasks_per_batch * total_n_support, n_way)
        support_labels_one_hot = support_labels_one_hot.view(tasks_per_batch, total_n_support, n_way)
        support_labels_one_hot = support_labels_one_hot.reshape(tasks_per_batch, total_n_support * n_way)
        # (tasks_per_batch, total_n_support * n_way)

        G = block_kernel_matrix
        e = -1.0 * support_labels_one_hot
        #This part is for the inequality constraints:
        #\alpha^m_i <= C^m_i \forall m,i
        #where C^m_i = C if m  = y_i,
        #C^m_i = 0 if m != y_i.
        id_matrix_1 = torch.eye(n_way * total_n_support).expand(tasks_per_batch, n_way * total_n_support, n_way * total_n_support)
        C = Variable(id_matrix_1)
        h = Variable(C_reg * support_labels_one_hot)
        #print (C.size(), h.size())
        #This part is for the equality constraints:
        #\sum_m \alpha^m_i=0 \forall i
        id_matrix_2 = torch.eye(total_n_support).expand(tasks_per_batch, total_n_support, total_n_support).cuda()

        A = Variable(batched_kronecker(id_matrix_2, torch.ones(tasks_per_batch, 1, n_way).cuda()))
        b = Variable(torch.zeros(tasks_per_batch, total_n_support))

        if self._double_precision:
            G, e, C, h, A, b = [x.double().cuda() for x in [G, e, C, h, A, b]]
        else:
            G, e, C, h, A, b = [x.float().cuda() for x in [G, e, C, h, A, b]]

        # Solve the following QP to fit SVM:
        #        \hat z =   argmin_z 1/2 z^T G z + e^T z
        #                 subject to Cz <= h
        # We use detach() to prevent backpropagation to fixed variables.
        qp_sol = QPFunction(verbose=False, maxIter=maxIter)(G, e.detach(), C.detach(), h.detach(), A.detach(), b.detach())
        # G is not detached, that is the only one that needs gradients, since its a function of phi(x).

        qp_sol = qp_sol.reshape(tasks_per_batch, total_n_support, n_way)

        if return_estimator:
            return torch.bmm(qp_sol.float().transpose(1 ,2), support)

        
        # Compute the classification score for query.
        compatibility_query = computeGramMatrix(support, query)
        compatibility_query = compatibility_query.float()
        compatibility_query = compatibility_query.unsqueeze(3).expand(tasks_per_batch, total_n_support, total_n_query, n_way)
        logits_query = qp_sol.float().unsqueeze(2).expand(tasks_per_batch, total_n_support, total_n_query, n_way)
        logits_query = logits_query * compatibility_query
        logits_query = torch.sum(logits_query, 1)

        # Compute the classification score for support.
        compatibility_support = computeGramMatrix(support, support)
        compatibility_support = compatibility_support.float()
        compatibility_support = compatibility_support.unsqueeze(3).expand(tasks_per_batch, total_n_support, total_n_support, n_way)
        logits_support = qp_sol.float().unsqueeze(2).expand(tasks_per_batch, total_n_support, total_n_support, n_way)
        logits_support = logits_support * compatibility_support
        logits_support = torch.sum(logits_support, 1)
        
        # compute loss and acc on support
        logits_support = logits_support.reshape(-1, logits_support.size(-1))
        labels_support = support_labels.reshape(-1)
        
        loss = self._inner_loss_func(logits_support, labels_support)
        accu = accuracy(logits_support, labels_support)
        measurements_trajectory['loss'].append(loss.item())
        measurements_trajectory['accu'].append(accu)


        return logits_query, measurements_trajectory

    
    def to(self, device, **kwargs):
        self._device = device
        self._model.to(device, **kwargs)
        

    def state_dict(self):
        # for model saving and reloading
        return {'model': self._model.state_dict()}


class ProtoNet(Algorithm):

    def __init__(self, model, inner_loss_func, device, 
            n_way, n_shot, n_query, normalize=True):
        
        self._model = model
        self._device = device
        self._inner_loss_func = inner_loss_func
        self._n_way = n_way
        self._n_shot = n_shot
        self._n_query = n_query
        self._normalize = normalize
        self.to(self._device)

    def euclidean_metric(self, a, b):

        print(a.shape, b.shape)
        n = a.shape[0]
        m = b.shape[0]
        d = b.shape[1]
        a = a.unsqueeze(1).expand(n, m, -1)
        b = b.unsqueeze(0).expand(n, m, -1)
        logits = -((a - b)**2).sum(dim=2)
        return logits
   
    def inner_loop_adapt(self, support, support_labels, query=None, return_estimator=False, scale=1.):
        """
        Constructs the prototype representation of each class(=mean of support vectors of each class) and 
        returns the classification score (=L2 distance to each class prototype) on the query set.
        
        This model is the classification head described in:
        Prototypical Networks for Few-shot Learning
        (Snell et al., NIPS 2017).
        
        Parameters:
        query:  a (n_tasks_per_batch, n_query, c, h, w) Tensor.
        support:  a (n_tasks_per_batch, n_support, c, h, w) Tensor.
        support_labels: a (n_tasks_per_batch, n_support) Tensor.
        n_way: a scalar. Represents the number of classes in a few-shot classification task.
        n_shot: a scalar. Represents the number of support examples given per class.
        normalize: a boolean. Represents whether if we want to normalize the distances by the embedding dimension.
        Returns: a (tasks_per_batch, n_query, n_way) Tensor.
        """

        measurements_trajectory = defaultdict(list)

        assert(query.dim() == 5)
        assert(support.dim() == 5)
        
        # get features
        orig_query_shape = query.shape
        orig_support_shape = support.shape
        # support = self._model(
        #     support.reshape(-1, *orig_support_shape[2:]), features_only=True).reshape(*orig_support_shape[:2], -1)
        # query = self._model(
        #     query.reshape(-1, *orig_query_shape[2:]), features_only=True).reshape(*orig_query_shape[:2], -1)

       

        # support = support.reshape(-1, *orig_support_shape[2:])
        # query = query.reshape(-1, *orig_query_shape[2:])
        # z = torch.cat([support, query], dim=0)
        # rand_ind = torch.randperm(z.shape[0])
        # inv_rand_ind = torch.randperm(z.shape[0])
        # for k in range(z.shape[0]):
        #     inv_rand_ind[rand_ind[k]] = k
        # z = self._model(z[rand_ind], features_only=True)[inv_rand_ind]
        # support = z[:support.shape[0]]
        # query = z[support.shape[0]:]
        # support = support.reshape(*orig_support_shape[:2], -1)
        # query = query.reshape(*orig_query_shape[:2], -1)
        
        support = self._model(
            support.reshape(-1, *orig_support_shape[2:]), features_only=True).reshape(*orig_support_shape[:2], -1)
        query = self._model(
            query.reshape(-1, *orig_query_shape[2:]), features_only=True).reshape(*orig_query_shape[:2], -1)
        

        tasks_per_batch = query.size(0)
        total_n_support = support.size(1) # support samples across all classes in a task
        total_n_query = query.size(1)     # query samples across all classes in a task
        d = query.size(2)                 # dimension

        n_way = self._n_way               # n_classes in a task
        n_query = self._n_query           # n_query samples per class
        n_shot = self._n_shot             # n_support samples per class
        normalize = self._normalize

        assert(query.dim() == 3)
        assert(support.dim() == 3)
        assert(query.size(0) == support.size(0) and query.size(2) == support.size(2))
        assert(total_n_support == n_way * n_shot)
        assert(total_n_query == n_way * n_query)

        support_labels_one_hot = one_hot(support_labels.view(tasks_per_batch * total_n_support), n_way)
        support_labels_one_hot = support_labels_one_hot.view(tasks_per_batch, total_n_support, n_way)
    
        labels_train_transposed = support_labels_one_hot.transpose(1,2)
        # this makes it tasks_per_batch x n_way x total_n_support

        prototypes = torch.bmm(labels_train_transposed, support)
        # [batch_size x n_way x d] =
        #     [batch_size x n_way x total_n_support] * [batch_size x total_n_support x d]

        prototypes = prototypes.div(
            labels_train_transposed.sum(dim=2, keepdim=True).expand_as(prototypes)
        )
        # Divide with the number of examples per novel category.

        if return_estimator:
            return prototypes

        ################################################
        # Compute the classification score for query
        ################################################

        # Distance Matrix Vectorization Trick
        AB = computeGramMatrix(query, prototypes)
        # batch_size x total_n_query x n_way
        AA = (query * query).sum(dim=2, keepdim=True)
        # batch_size x total_n_query x 1
        BB = (prototypes * prototypes).sum(dim=2, keepdim=True).reshape(tasks_per_batch, 1, n_way)
        # batch_size x 1 x n_way
        logits_query = AA.expand_as(AB) - 2 * AB + BB.expand_as(AB)
        # euclidean distance 
        logits_query = -logits_query
        # batch_size x total_n_query x n_way

        if normalize:
            logits_query = logits_query / d
        
        ################################################
        # Compute the classification score for query
        ################################################

        # Distance Matrix Vectorization Trick
        AB = computeGramMatrix(support, prototypes)
        # batch_size x total_n_support x n_way
        AA = (support * support).sum(dim=2, keepdim=True)
        # batch_size x total_n_support x 1
        ## BB needn't be computed again
        logits_support = AA.expand_as(AB) - 2 * AB + BB.expand_as(AB)
        # euclidean distance 
        logits_support = -logits_support
        # batch_size x total_n_support x n_way

        if normalize:
            logits_support = logits_support / d

        
        # compute loss and acc on support
        logits_support = logits_support.reshape(-1, logits_support.size(-1)) * scale


        # logits_query = self.euclidean_metric(self._model(query), proto) 
        # logits_support = self.euclidean_metric(sup_op, proto)  * scale
        

        labels_support = support_labels.reshape(-1)
        

        loss = self._inner_loss_func(logits_support, labels_support)
        accu = accuracy(logits_support, labels_support) * 100.
        measurements_trajectory['loss'].append(loss.item())
        measurements_trajectory['accu'].append(accu)

        return logits_query, measurements_trajectory

    def to(self, device, **kwargs):
        self._device = device
        self._model.to(device, **kwargs)

    def state_dict(self):
        # for model saving and reloading
        return {'model': self._model.state_dict()}






class Finetune(Algorithm):

    def __init__(self, model, inner_loss_func, device, 
        n_updates, classifier_type, final_feat_dim, n_way, aux_loss=None):
        
        self._model = model
        self._device = device
        self._inner_loss_func = inner_loss_func
        self._n_updates = n_updates
        self._aux_loss = aux_loss

        if classifier_type == 'linear':
            self._fc = nn.Linear(final_feat_dim, n_way)
        elif classifier_type == 'distance-classifier':
            self._fc = distLinear(final_feat_dim, n_way)
        elif classifier_type == 'gda':
            self._fc = gaussianDA(final_feat_dim, n_way)
        elif classifier_type == 'ortho-classifier':
            self._fc = orthonormalDistLinear(final_feat_dim, n_way)
        else:
            raise ValueError("classifier type not found")

        self._model.eval()
        self.to(self._device)
   
    def inner_loop_adapt(self, support, support_labels, query=None, return_estimator=False):
        """
        Finetunes last layer using GD.

        Parameters:
        query:  a (n_tasks_per_batch, n_query, c, h, w) Tensor.
        support:  a (n_tasks_per_batch, n_support, c, h, w) Tensor.
        support_labels: a (n_tasks_per_batch, n_support) Tensor.
        n_way: a scalar. Represents the number of classes in a few-shot classification task.
        n_shot: a scalar. Represents the number of support examples given per class.
        normalize: a boolean. Represents whether if we want to normalize the distances by the embedding dimension.
        Returns: a (tasks_per_batch, n_query, n_way) Tensor.
        """

        measurements_trajectory = defaultdict(list)

        assert(query.dim() == 5)
        assert(support.dim() == 5)
        

        n_tasks = query.size(0)
        assert support.size(0) == n_tasks
        assert support_labels.size(0) == n_tasks

        logits_query = []
        avg_loss = 0.
        abg_accu = 0.
        
        self._model.eval()
        self._fc.train()

        for i in range(n_tasks):

            task_support = support[i]
            task_query = query[i]
            task_support_labels = support_labels[i] 

            # save model dict
            # saved_model_dict = self._model.state_dict()

            # set optimizer
            optimizer = torch.optim.Adam(self._fc.parameters(),
                lr = 0.001, weight_decay=0.001)

            # optimize last layer
            for j in range(self._n_updates):
                optimizer.zero_grad()
                features = self._model(task_support, features_only=True)
                features = features.detach()
                logits = self._fc(features)
                loss = self._inner_loss_func(logits, task_support_labels)
                if self._aux_loss is not None and (j % 10 == 0):
                    loss += self._aux_loss(features, task_support_labels, self._fc)
                loss.backward()
                optimizer.step()
            
            # metrics
            avg_loss += loss.item()
            avg_accu = accuracy(logits, task_support_labels) * 100.

            # predict query logits
            task_logits_query = self._fc(self._model(task_query, features_only=True))
            logits_query.append(task_logits_query)

            # reset model dict
            # self._model.load_state_dict(saved_model_dict)

        logits_query = torch.cat(logits_query, dim=0)
        measurements_trajectory['loss'].append(avg_loss)
        measurements_trajectory['accu'].append(avg_accu)

        return logits_query, measurements_trajectory

    def to(self, device, **kwargs):
        self._device = device
        self._model.to(device, **kwargs)
        self._fc.to(device, **kwargs)

    def state_dict(self):
        # for model saving and reloading
        return {'model': self._model.state_dict()}





class ProtoCosineNet(Algorithm):

    def __init__(self, model, inner_loss_func, device, 
            n_way, n_shot, n_query):
        
        self._model = model
        self._device = device
        self._inner_loss_func = inner_loss_func
        self._n_way = n_way
        self._n_shot = n_shot
        self._n_query = n_query
        self.to(self._device)
   
    def inner_loop_adapt(self, support, support_labels, query, scale=1.):
        """
        Constructs the prototype representation of each class(=mean of support vectors of each class) and 
        returns the classification score (=cosine distance to each class prototype) on the query set.
        
        This model is the classification head described in:
        Prototypical Networks for Few-shot Learning
        (Snell et al., NIPS 2017).
        
        Parameters:
        query:  a (n_tasks_per_batch, n_query, c, h, w) Tensor.
        support:  a (n_tasks_per_batch, n_support, c, h, w) Tensor.
        support_labels: a (n_tasks_per_batch, n_support) Tensor.
        Returns: a (tasks_per_batch, n_query, n_way) Tensor.
        """

        measurements_trajectory = defaultdict(list)

        assert(query.dim() == 5)
        assert(support.dim() == 5)
        
        # get features
        orig_query_shape = query.shape
        orig_support_shape = support.shape
        support = self._model(
            support.reshape(-1, *orig_support_shape[2:]), features_only=True).reshape(*orig_support_shape[:2], -1)
        query = self._model(
            query.reshape(-1, *orig_query_shape[2:]), features_only=True).reshape(*orig_query_shape[:2], -1)


        # support = support.div(torch.norm(support, dim=2, keepdim=True)+0.00001)
        # query = query.div(torch.norm(query, dim=2, keepdim=True)+0.00001)
        # Project onto hyper-sphere.

        
        tasks_per_batch = query.size(0)
        total_n_support = support.size(1) # support samples across all classes in a task
        total_n_query = query.size(1)     # query samples across all classes in a task
        d = query.size(2)                 # dimension

        n_way = self._n_way               # n_classes in a task
        n_query = self._n_query           # n_query samples per class
        n_shot = self._n_shot             # n_support samples per class
        
        assert(query.dim() == 3)
        assert(support.dim() == 3)
        assert(query.size(0) == support.size(0) and query.size(2) == support.size(2))
        assert(total_n_support == n_way * n_shot)
        assert(total_n_query == n_way * n_query)

        support_labels_one_hot = one_hot(support_labels.view(tasks_per_batch * total_n_support), n_way)
        support_labels_one_hot = support_labels_one_hot.view(tasks_per_batch, total_n_support, n_way)
    
        labels_train_transposed = support_labels_one_hot.transpose(1,2)
        # this makes it tasks_per_batch x n_way x total_n_support

        prototypes = torch.bmm(labels_train_transposed, support)
        # [batch_size x n_way x d] =
        #     [batch_size x n_way x total_n_support] * [batch_size x total_n_support x d]

        prototypes = prototypes.div(
            labels_train_transposed.sum(dim=2, keepdim=True).expand_as(prototypes)
        )
        # Divide with the number of examples per base category.
        # prototypes = prototypes.div(torch.norm(prototypes, dim=2, keepdim=True)+0.00001)
        # Project onto hyper-sphere.

        
        ################################################
        # Compute the classification score for query
        ################################################

        # Distance Matrix Vectorization Trick
        logits_query = computeGramMatrix(query, prototypes)
        # batch_size x total_n_query x n_way
        
        ################################################
        # Compute the classification score for query
        ################################################

        # Distance Matrix Vectorization Trick
        logits_support = computeGramMatrix(support, prototypes)
        # batch_size x total_n_support x n_way

        # compute loss and acc on support
        logits_support = logits_support.reshape(-1, logits_support.size(-1)) * scale
        labels_support = support_labels.reshape(-1)
        
        loss = self._inner_loss_func(logits_support, labels_support)
        accu = accuracy(logits_support, labels_support) * 100.
        measurements_trajectory['loss'].append(loss.item())
        measurements_trajectory['accu'].append(accu)

        return logits_query, measurements_trajectory

    def to(self, device, **kwargs):
        self._device = device
        self._model.to(device, **kwargs)

    def state_dict(self):
        # for model saving and reloading
        return {'model': self._model.state_dict()}
















class ProtoCosineNetCorrected2(Algorithm):

    def __init__(self, model, inner_loss_func, device, 
            n_way, n_shot, n_query):
        
        self._model = model
        self._device = device
        self._inner_loss_func = inner_loss_func
        self._n_way = n_way
        self._n_shot = n_shot
        self._n_query = n_query
        self.to(self._device)
   
    def inner_loop_adapt(self, support, support_labels, query):
        """
        Constructs the prototype representation of each class(=mean of support vectors of each class) and 
        returns the classification score (=cosine distance to each class prototype) on the query set.
        
        This model is the classification head described in:
        Prototypical Networks for Few-shot Learning
        (Snell et al., NIPS 2017).
        
        Parameters:
        query:  a (n_tasks_per_batch, n_query, c, h, w) Tensor.
        support:  a (n_tasks_per_batch, n_support, c, h, w) Tensor.
        support_labels: a (n_tasks_per_batch, n_support) Tensor.
        Returns: a (tasks_per_batch, n_query, n_way) Tensor.
        """

        measurements_trajectory = defaultdict(list)

        assert(query.dim() == 5)
        assert(support.dim() == 5)
        
        # get features
        orig_query_shape = query.shape
        orig_support_shape = support.shape
        support = self._model(
            support.reshape(-1, *orig_support_shape[2:]), features_only=True).reshape(*orig_support_shape[:2], -1)
        query = self._model(
            query.reshape(-1, *orig_query_shape[2:]), features_only=True).reshape(*orig_query_shape[:2], -1)


        # support = support.div(torch.norm(support, dim=2, keepdim=True)+0.00001)
        # query = query.div(torch.norm(query, dim=2, keepdim=True)+0.00001)
        # Project onto hyper-sphere.

        
        tasks_per_batch = query.size(0)
        total_n_support = support.size(1) # support samples across all classes in a task
        total_n_query = query.size(1)     # query samples across all classes in a task
        d = query.size(2)                 # dimension

        n_way = self._n_way               # n_classes in a task
        n_query = self._n_query           # n_query samples per class
        n_shot = self._n_shot             # n_support samples per class
        
        assert(query.dim() == 3)
        assert(support.dim() == 3)
        assert(query.size(0) == support.size(0) and query.size(2) == support.size(2))
        assert(total_n_support == n_way * n_shot)
        assert(total_n_query == n_way * n_query)

        support_labels_one_hot = one_hot(support_labels.view(tasks_per_batch * total_n_support), n_way)
        support_labels_one_hot = support_labels_one_hot.view(tasks_per_batch, total_n_support, n_way)
    
        labels_train_transposed = support_labels_one_hot.transpose(1,2)
        # this makes it tasks_per_batch x n_way x total_n_support

        prototypes = torch.bmm(labels_train_transposed, support)
        # [batch_size x n_way x d] =
        #     [batch_size x n_way x total_n_support] * [batch_size x total_n_support x d]

        prototypes = prototypes.div(
            labels_train_transposed.sum(dim=2, keepdim=True).expand_as(prototypes)
        )
        prototypes = (n_way) / (n_way - 1) * (prototypes - prototypes.mean(dim=1, keepdim=True))
        # Divide with the number of examples per base category.
        # prototypes = prototypes.div(torch.norm(prototypes, dim=2, keepdim=True)+0.00001)
        # Project onto hyper-sphere.

        
        ################################################
        # Compute the classification score for query
        ################################################

        # Distance Matrix Vectorization Trick
        logits_query = computeGramMatrix(query, prototypes)
        # batch_size x total_n_query x n_way
        
        ################################################
        # Compute the classification score for query
        ################################################

        # Distance Matrix Vectorization Trick
        logits_support = computeGramMatrix(support, prototypes)
        # batch_size x total_n_support x n_way

        # compute loss and acc on support
        logits_support = logits_support.reshape(-1, logits_support.size(-1))
        labels_support = support_labels.reshape(-1)
        
        loss = self._inner_loss_func(logits_support, labels_support)
        accu = accuracy(logits_support, labels_support) * 100.
        measurements_trajectory['loss'].append(loss.item())
        measurements_trajectory['accu'].append(accu)

        return logits_query, measurements_trajectory

    def to(self, device, **kwargs):
        self._device = device
        self._model.to(device, **kwargs)

    def state_dict(self):
        # for model saving and reloading
        return {'model': self._model.state_dict()}











class Protomax(Algorithm):

    def __init__(self, model, inner_loss_func, device, 
            n_way, n_shot, n_query, normalize=True):
        
        self._model = model
        self._device = device
        self._inner_loss_func = inner_loss_func
        self._n_way = n_way
        self._n_shot = n_shot
        self._n_query = n_query
        self._normalize = normalize
        self.to(self._device)

    def euclidean_metric(self, a, b):

        print(a.shape, b.shape)
        n = a.shape[0]
        m = b.shape[0]
        d = b.shape[1]
        a = a.unsqueeze(1).expand(n, m, -1)
        b = b.unsqueeze(0).expand(n, m, -1)
        logits = -((a - b)**2).sum(dim=2)
        return logits
   
    def inner_loop_adapt(self, support, support_labels, query=None, return_estimator=False, scale=1.):
        """
        Constructs the prototype representation of each class(=mean of support vectors of each class) and 
        returns the classification score (=L2 distance to each class prototype) on the query set.
        
        This model is the classification head described in:
        Prototypical Networks for Few-shot Learning
        (Snell et al., NIPS 2017).
        
        Parameters:
        query:  a (n_tasks_per_batch, n_query, c, h, w) Tensor.
        support:  a (n_tasks_per_batch, n_support, c, h, w) Tensor.
        support_labels: a (n_tasks_per_batch, n_support) Tensor.
        n_way: a scalar. Represents the number of classes in a few-shot classification task.
        n_shot: a scalar. Represents the number of support examples given per class.
        normalize: a boolean. Represents whether if we want to normalize the distances by the embedding dimension.
        Returns: a (tasks_per_batch, n_query, n_way) Tensor.
        """

        measurements_trajectory = defaultdict(list)

        assert(query.dim() == 5)
        assert(support.dim() == 5)
        
        # get features
        orig_query_shape = query.shape
        orig_support_shape = support.shape
        
        
        support = self._model(
            support.reshape(-1, *orig_support_shape[2:]), features_only=True).reshape(*orig_support_shape[:2], -1)
        query = self._model(
            query.reshape(-1, *orig_query_shape[2:]), features_only=True).reshape(*orig_query_shape[:2], -1)
        

        tasks_per_batch = query.size(0)
        total_n_support = support.size(1) # support samples across all classes in a task
        total_n_query = query.size(1)     # query samples across all classes in a task
        d = query.size(2)                 # dimension

        n_way = self._n_way               # n_classes in a task
        n_query = self._n_query           # n_query samples per class
        n_shot = self._n_shot             # n_support samples per class
        normalize = self._normalize

        assert(query.dim() == 3)
        assert(support.dim() == 3)
        assert(query.size(0) == support.size(0) and query.size(2) == support.size(2))
        assert(total_n_support == n_way * n_shot)
        assert(total_n_query == n_way * n_query)
        
        labels_support = support_labels.reshape(-1)

        # support_labels_one_hot = one_hot(support_labels.view(tasks_per_batch * total_n_support), n_way)
        # support_labels_one_hot = support_labels_one_hot.view(tasks_per_batch, total_n_support, n_way)
    
        # labels_train_transposed = support_labels_one_hot.transpose(1,2)
        # this makes it tasks_per_batch x n_way x total_n_support

        # prototypes = torch.bmm(labels_train_transposed, support)
        # [batch_size x n_way x d] =
        #     [batch_size x n_way x total_n_support] * [batch_size x total_n_support x d]

        # if return_estimator:
        #     return prototypes

        ################################################
        # Compute the classification score for query
        ################################################

        # Distance Matrix Vectorization Trick
        AB = computeGramMatrix(query, support)
        # batch_size x total_n_query x total_n_support
        AA = (query * query).sum(dim=2, keepdim=True)
        # batch_size x total_n_query x 1
        BB = (support * support).sum(dim=2, keepdim=True).reshape(tasks_per_batch, 1, total_n_support)
        # batch_size x 1 x total_n_support
        logits_query = AA.expand_as(AB) - 2 * AB + BB.expand_as(AB)
        # euclidean distance 
        logits_query = -logits_query
        # batch_size x total_n_query x total_n_support
        logits_query = logits_query.reshape(tasks_per_batch, total_n_query, n_way, n_shot)
        logits_query = torch.max(logits_query, dim=3)[0]
        # batch_size x total_n_query x n_way
        
        if normalize:
            logits_query = logits_query / d

        # fix label order        
        lst = [k for k, g in itertools.groupby(labels_support.cpu().numpy())]
        rev_lst = [0] * len(lst)
        for j, label in enumerate(lst):
            rev_lst[label] = j
        logits_query = logits_query[:, :, rev_lst]
        
        ################################################
        # Compute the classification score for query
        ################################################

        # Distance Matrix Vectorization Trick
        AB = computeGramMatrix(support, support)
        # batch_size x total_n_support x total_n_support
        AA = (support * support).sum(dim=2, keepdim=True)
        # batch_size x total_n_support x 1
        ## BB needn't be computed again
        logits_support = AA.expand_as(AB) - 2 * AB + BB.expand_as(AB)
        # euclidean distance 
        logits_support = -logits_support
        # batch_size x total_n_support x total_n_support
        logits_support = logits_support.reshape(tasks_per_batch, total_n_support, n_way, n_shot)
        logits_support = torch.max(logits_support, dim=3)[0]
        # batch_size x total_n_support x n_way
        
        if normalize:
            logits_support = logits_support / d
        
        
        # fix label order        
        # print(logits_support.shape, rev_lst)
        logits_support = logits_support[:, :, rev_lst]
        # print(logits_support.shape)
        
        # print("logits_support max", torch.max(logits_support, dim=-1)[1])
        # print("labels_support", labels_support)
        
        # compute loss and acc on support
        logits_support = logits_support.reshape(-1, logits_support.size(-1)) * scale
        

        loss = self._inner_loss_func(logits_support, labels_support)
        accu = accuracy(logits_support, labels_support) * 100.
        measurements_trajectory['loss'].append(loss.item())
        measurements_trajectory['accu'].append(accu)

        return logits_query, measurements_trajectory

    def to(self, device, **kwargs):
        self._device = device
        self._model.to(device, **kwargs)

    def state_dict(self):
        # for model saving and reloading
        return {'model': self._model.state_dict()}



class Ridge(Algorithm):

    def __init__(self, model, inner_loss_func, device, 
            n_way, n_shot, n_query, normalize=True):
        
        self._model = model
        self._device = device
        self._inner_loss_func = inner_loss_func
        self._n_way = n_way
        self._n_shot = n_shot
        self._n_query = n_query
        self._normalize = normalize
        self._lambda_reg = 50.0
        self._double_precision = False
        self.to(self._device)


    def inner_loop_adapt(self, support, support_labels, query=None, return_estimator=False, scale=1.):

        """
        Fits the support set with ridge regression and 
        returns the classification score on the query set.
        Parameters:
        query:  a (n_tasks_per_batch, n_query, c, h, w) Tensor.
        support:  a (n_tasks_per_batch, n_support, c, h, w) Tensor.
        support_labels: a (tasks_per_batch, n_support) Tensor.
        lambda_reg: a scalar. Represents the strength of L2 regularization.
        Returns: a (tasks_per_batch, n_query, n_way) Tensor.
        """


        measurements_trajectory = defaultdict(list)

        assert(query.dim() == 5)
        assert(support.dim() == 5)
        
        # get features
        orig_query_shape = query.shape
        orig_support_shape = support.shape
        
        support = self._model(
            support.reshape(-1, *orig_support_shape[2:]), features_only=True).reshape(*orig_support_shape[:2], -1)
        query = self._model(
            query.reshape(-1, *orig_query_shape[2:]), features_only=True).reshape(*orig_query_shape[:2], -1)
        
        
        lambda_reg = self._lambda_reg
        double_precision = self._double_precision
        tasks_per_batch = query.size(0)
        n_support = support.size(1)
        n_query = query.size(1)
        n_way = self._n_way
        n_shot = self._n_shot

        assert(query.dim() == 3)
        assert(support.dim() == 3)
        assert(query.size(0) == support.size(0) and query.size(2) == support.size(2))
        assert(n_support == n_way * n_shot)      # n_support must equal to n_way * n_shot

        #Here we solve the dual problem:
        #Note that the classes are indexed by m & samples are indexed by i.
        #min_{\alpha}  0.5 \sum_m ||w_m(\alpha)||^2 + \sum_i \sum_m e^m_i alpha^m_i

        #where w_m(\alpha) = \sum_i \alpha^m_i x_i,
        
        #\alpha is an (n_support, n_way) matrix
        kernel_matrix = computeGramMatrix(support, support)
        kernel_matrix += lambda_reg * torch.eye(n_support).expand(tasks_per_batch, n_support, n_support).cuda()

        block_kernel_matrix = kernel_matrix.repeat(n_way, 1, 1) #(n_way * tasks_per_batch, n_support, n_support)
        
        support_labels_one_hot = one_hot(support_labels.view(tasks_per_batch * n_support), n_way) # (tasks_per_batch * n_support, n_way)
        support_labels_one_hot = support_labels_one_hot.transpose(0, 1) # (n_way, tasks_per_batch * n_support)
        support_labels_one_hot = support_labels_one_hot.reshape(n_way * tasks_per_batch, n_support)     # (n_way*tasks_per_batch, n_support)
        
        G = block_kernel_matrix
        e = -2.0 * support_labels_one_hot
        
        #This is a fake inequlity constraint as qpth does not support QP without an inequality constraint.
        id_matrix_1 = torch.zeros(tasks_per_batch*n_way, n_support, n_support)
        C = Variable(id_matrix_1)
        h = Variable(torch.zeros((tasks_per_batch*n_way, n_support)))
        dummy = Variable(torch.Tensor()).cuda()      # We want to ignore the equality constraint.

        if double_precision:
            G, e, C, h = [x.double().cuda() for x in [G, e, C, h]]

        else:
            G, e, C, h = [x.float().cuda() for x in [G, e, C, h]]

        # Solve the following QP to fit SVM:
        #        \hat z =   argmin_z 1/2 z^T G z + e^T z
        #                 subject to Cz <= h
        # We use detach() to prevent backpropagation to fixed variables.
        qp_sol = QPFunction(verbose=False)(G, e.detach(), C.detach(), h.detach(), dummy.detach(), dummy.detach())
        #qp_sol = QPFunction(verbose=False)(G, e.detach(), dummy.detach(), dummy.detach(), dummy.detach(), dummy.detach())

        #qp_sol (n_way*tasks_per_batch, n_support)
        qp_sol = qp_sol.reshape(n_way, tasks_per_batch, n_support)
        #qp_sol (n_way, tasks_per_batch, n_support)
        qp_sol = qp_sol.permute(1, 2, 0)
        #qp_sol (tasks_per_batch, n_support, n_way)
        
        # Compute the classification score.
        compatibility = computeGramMatrix(support, query)
        compatibility = compatibility.float()
        compatibility = compatibility.unsqueeze(3).expand(tasks_per_batch, n_support, n_query, n_way)
        qp_sol = qp_sol.reshape(tasks_per_batch, n_support, n_way)
        logits = qp_sol.float().unsqueeze(2).expand(tasks_per_batch, n_support, n_query, n_way)
        logits = logits * compatibility
        logits = torch.sum(logits, 1)

        # compute loss and acc on support
        with torch.no_grad():
            compatibility = computeGramMatrix(support, support)
            compatibility = compatibility.float()
            compatibility = compatibility.unsqueeze(3).expand(tasks_per_batch, n_support, n_support, n_way)
            logits_support = qp_sol.float().unsqueeze(2).expand(tasks_per_batch, n_support, n_support, n_way)
            logits_support = logits_support * compatibility
            logits_support = torch.sum(logits_support, 1)
            logits_support = logits_support.reshape(-1, logits_support.size(-1)) * scale
            loss = self._inner_loss_func(logits_support, support_labels.reshape(-1))
            accu = accuracy(logits_support, support_labels.reshape(-1)) * 100.
            measurements_trajectory['loss'].append(loss.item())
            measurements_trajectory['accu'].append(accu)

        return logits, measurements_trajectory

    def to(self, device, **kwargs):
        self._device = device
        self._model.to(device, **kwargs)

    def state_dict(self):
        # for model saving and reloading
        return {'model': self._model.state_dict()}
