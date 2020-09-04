"""
    The calculation to be performed at each point (modified model), evaluating
    the loss value, accuracy and eigen values of the hessian matrix
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.autograd.variable import Variable
from collections import defaultdict
import numpy as np


def get_features(model, dataloader):
    # iterator = tqdm(enumerate(dataloader, start=1),
    #                     leave=False, file=sys.stdout, position=0)
    iterator = iter(dataloader)
    features = defaultdict(list)
    model.eval()
    for batch in iterator:
        batch_x, batch_y = batch
        batch_x = batch_x.cuda()
        batch_y = batch_y.cpu().numpy()
        features_x = model(batch_x, features_only=True)
        for y in np.unique(batch_y):
            features[y].append(features_x[batch_y==y].detach().cpu().numpy())
    for key in features:
        features[key] = np.concatenate(features[key], axis=0) 
    return features



def evaluate_ineq(a, b):
    diff = a - b
    diff = diff / np.linalg.norm(diff, axis=1)[:, None]
    num = 1.0 - (diff[0, :].T @ diff[1, :])
    deno = 1.0 - (a[0, :] @ b[0, :].T) + 1.0 - (a[1, :] @ b[1, :].T)
    return num / deno

def hyperplane_variance(features):
    X = np.concatenate([features[x] for x in sorted(features)], axis=0)
    y = np.concatenate([[z] * len(features[z]) for z in np.arange(len(features))], axis=0)
    all_labels = set(y)
    n_runs = 100
    r_hv = []
    for _ in range(n_runs):
        binary_problem_labels = np.random.choice(
            list(all_labels), 2, replace=False)
        X_1 = X[y==binary_problem_labels[0], :]
        X_2 = X[y==binary_problem_labels[1], :]
        rhv_pair_classes = 0.
        n_inner_runs = 50
        for _ in range(n_inner_runs):
            random_indices_1 = np.random.choice(len(X_1), 2, replace=False)
            random_indices_2 = np.random.choice(len(X_2), 2, replace=False)
            rhv_pair_classes += evaluate_ineq(X_1[random_indices_1, :], X_2[random_indices_2, :])
        r_hv.append(rhv_pair_classes / n_inner_runs)
    return np.mean(r_hv) 


def eval_loss(net, criterion, loader, use_cuda=True, aux_loader=None):
    """
    Evaluate the loss value for a given 'net' on the dataset provided by the loader.

    Args:
        net: the neural net model
        criterion: loss function
        loader: dataloader
        use_cuda: use cuda or not
    Returns:
        loss value and accuracy
    """
    correct = 0
    total_loss = 0
    total = 0 # number of samples
    num_batch = len(loader)

    if use_cuda:
        net.cuda()
    net.eval()

    with torch.no_grad():
        features = get_features(net, loader)
        rhv = hyperplane_variance(features)
    return rhv, 100.

    # loader = iter(loader)
    # aux_iterator = None
    # if aux_loader is not None:
    #     aux_iterator = iter(aux_loader)
    # with torch.no_grad():
    #     if isinstance(criterion, nn.CrossEntropyLoss):
    #         for batch_idx, (inputs, targets) in enumerate(loader):
    #             # print("batch_idx", batch_idx, "total", total)
                
    #             if aux_iterator is not None:
    #                 aux_batch_x, aux_batch_y = next(aux_iterator)
    #                 aux_batch_x = aux_batch_x.cuda()
    #                 aux_batch_y = aux_batch_y.cuda()
    #                 aux_features_x = net(aux_batch_x, features_only=True)
    #                 net.fc.update_L(aux_features_x, aux_batch_y)
                
    #             batch_size = inputs.size(0)
    #             total += batch_size
    #             inputs = Variable(inputs)
    #             targets = Variable(targets)
    #             if use_cuda:
    #                 inputs, targets = inputs.cuda(), targets.cuda()
    #             outputs = net(inputs)
    #             loss = criterion(outputs, targets)
    #             total_loss += loss.item()*batch_size
    #             _, predicted = torch.max(outputs.data, 1)
    #             correct += predicted.eq(targets).sum().item()

    #     elif isinstance(criterion, nn.MSELoss):
    #         for batch_idx, (inputs, targets) in enumerate(loader):
    #             batch_size = inputs.size(0)
    #             total += batch_size
    #             inputs = Variable(inputs)

    #             one_hot_targets = torch.FloatTensor(batch_size, 10).zero_()
    #             one_hot_targets = one_hot_targets.scatter_(1, targets.view(batch_size, 1), 1.0)
    #             one_hot_targets = one_hot_targets.float()
    #             one_hot_targets = Variable(one_hot_targets)
    #             if use_cuda:
    #                 inputs, one_hot_targets = inputs.cuda(), one_hot_targets.cuda()
    #             outputs = F.softmax(net(inputs))
    #             loss = criterion(outputs, one_hot_targets)
    #             total_loss += loss.item()*batch_size
    #             _, predicted = torch.max(outputs.data, 1)
    #             correct += predicted.cpu().eq(targets).sum().item()

    # return total_loss/total, 100.*correct/total
