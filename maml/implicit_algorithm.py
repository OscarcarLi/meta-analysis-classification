import cvxpy as cp 
from cvxpylayers.torch import CvxpyLayer 
import torch  
from collections import defaultdict, OrderedDict
from maml.grad import soft_clip, get_grad_norm, get_grad_quantiles
from maml.utils import accuracy
from maml.logistic_regression_utils import logistic_regression_hessian_with_respect_to_w, logistic_regression_mixed_derivatives_with_respect_to_w_then_to_X
from maml.utils import spectral_norm
import torch.nn.functional as F
import numpy as np
from maml.models.lstm_embedding_model import LSTMAttentionEmbeddingModel
from sklearn.linear_model import LogisticRegression
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter("always", ConvergenceWarning)



class Algorithm(object):
    def __init__(self):
        pass

    def inner_loop_adapt(self, task, num_updates=None, analysis=False, iter=None):
        raise NotImplementedError('Subclass should implement inner_loop')

    def predict_without_adapt(self, train_task, batch, param_dict=None):
        raise NotImplementedError()


class ImpRMAML_inner_algorithm(Algorithm):
    def __init__(self, model, embedding_model,
                inner_loss_func, l2_lambda,
                device, is_classification=True):

        self._model = model
        self._embedding_model = embedding_model
        self._inner_loss_func = inner_loss_func
        self._l2_lambda = l2_lambda
        self._device = device
        self._is_classification = is_classification
        self._C = 1./self._l2_lambda
        self.to(self._device)

    def inner_loop_adapt(self, task, num_updates=None, analysis=False, iter=None):
        # adapt means doing the complete inner loop update
        
        measurements_trajectory = defaultdict(list)

        modulation, _ = self._embedding_model(task, return_task_embedding=False)

        # here the features are padded with 1's at the end
        features_X = self._model(
            task.x, modulation=modulation)

        # cvxpy variables
        features = torch.autograd.Variable(
            features_X.detach().cpu(), requires_grad=True)
        y = (task.y).cpu().numpy()
        n_classes = max(y) + 1
        n_samples = features.shape[0]
        
        W = cp.Variable((n_classes, self._model._modulation_mat_rank+1)) 
        xi = cp.Variable((n_samples, n_classes))
        X_cp = cp.Parameter(features.shape)
        constraints = [xi >= 0]
        z = X_cp @ W.T
        for i in range(n_samples):
            for j in range(n_classes):
                if y[i] != j:
                    constraints.append(z[i][y[i]] >= 1 + z[i][j] - xi[i][j])
        obj = cp.Minimize(0.5 * cp.sum_squares(W) + self._C * cp.sum(xi))
        problem = cp.Problem(obj, constraints) 

        # create a diff convex layer
        cvxpylayer = CvxpyLayer(problem, parameters=[X_cp], variables=[W, xi])
        # solve the problem
        with warnings.catch_warnings(record=True) as wn:
            warnings.simplefilter("ignore")
            adapted_params, dual_var = cvxpylayer(features)
            adapted_params = adapted_params.to(self._device)

        with torch.no_grad():    
            preds = F.linear(features_X, weight=adapted_params)
            loss = self._inner_loss_func(preds, task.y)
            
        measurements_trajectory['loss'].append(loss.item())
        if self._is_classification: 
            measurements_trajectory['accu'].append(accuracy(preds, task.y))

        info_dict = None
        return adapted_params, features, modulation, measurements_trajectory, info_dict


    def to(self, device, **kwargs):
        # called in __init__
        self._device = device
        self._model.to(device, **kwargs)
        self._embedding_model.to(device, **kwargs)


    def state_dict(self):
        # for model saving and reloading
        return {'model': self._model.state_dict(),
                'embedding_model': self._embedding_model.state_dict()}





def test():
    
    # torch stuff
    x = torch.autograd.Variable(torch.randn(5, 1600), requires_grad=True)
    A = torch.autograd.Variable(torch.randn(1600, 5), requires_grad=True)
    y = torch.arange(5).numpy()
    # modulated_x = torch.cat([x @ A, torch.ones((x.shape[0], 1), device=x.device)], dim=-1) 
    modulated_x = x @ A
    n_classes = len(y)
    n_samples = x.shape[0]

    # cvxpy variables
    C = .01
    W = cp.Variable((5, 5)) 
    xi = cp.Variable((5, 5))
    x_cp = cp.Parameter(modulated_x.shape)
    constraints = [xi >= 0]
    z = x_cp @ W.T
    for i in range(n_samples):
        for j in range(n_classes):
            if y[i] != j:
                constraints.append(z[i][y[i]] >= 1 + z[i][j] - xi[i][j])
    obj = cp.Minimize(0.5 * cp.sum_squares(W) + C* cp.sum(xi))
    problem = cp.Problem(obj, constraints) 

    # check for dpp
    # print(problem.is_dpp())
    # print(problem.is_dpp())

    # create a diff convex layer
    cvxpylayer = CvxpyLayer(problem, parameters=[x_cp], variables=[W, xi])
    cvxpylayer = cvxpylayer.cuda()

    # solve the problem
    with warnings.catch_warnings():
        # warnings.simplefilter("ignore")
        solution = cvxpylayer(modulated_x)
    
    # print(solution[0].shape, solution[1].shape)
    # print(type(solution[0]))
    wt = solution[0].cuda()
    modulated_x = modulated_x.cuda()
    preds = F.linear(modulated_x, weight=wt, bias=None)
    # print(torch.argmax(F.linear(modulated_x, weight=solution[0], bias=None), dim=-1))
    # compute the gradient wrt parameters
    loss_layer = torch.nn.MultiLabelMarginLoss()
    loss = loss_layer(preds, torch.eye(5, dtype=torch.long, device=preds.device))
    # print(loss)
    loss.backward()
    # print(x.grad)
 


if __name__ == '__main__':
    test()
