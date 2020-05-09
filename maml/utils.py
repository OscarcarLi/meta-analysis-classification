import subprocess

import torch
import torch.nn.functional as F


def accuracy(preds, y):
    _, preds = torch.max(preds.data, 1)
    total = y.size(0)
    correct = (preds == y).sum().float()
    return (correct / total).item()


def spectral_norm(weight_mat, limit=10., n_power_iterations=2, eps=1e-12, device='cpu'):
    h, w = weight_mat.size()
    # randomly initialize `u` and `v`
    u = F.normalize(torch.randn(h), dim=0, eps=eps).to(device)
    v = F.normalize(torch.randn(w), dim=0, eps=eps).to(device)
    with torch.no_grad():
        for _ in range(n_power_iterations):
            # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
            # are the first left and right singular vectors.
            # This power iteration produces approximations of `u` and `v`.
            v = F.normalize(
                torch.mv(weight_mat.t(), u), dim=0, eps=eps, out=v)
            u = F.normalize(
                torch.mv(weight_mat, v), dim=0, eps=eps, out=u)   
        sigma = torch.dot(u, torch.mv(weight_mat, v))
    if sigma > limit:
        weight_mat = (weight_mat / sigma) * limit
    return weight_mat


def optimizer_to_device(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


def get_git_revision_hash():
    return str(subprocess.check_output(['git', 'rev-parse', 'HEAD']))


def primal_svm(preds, true_y, w, C, device):

    loss = 0.5 * w.pow(2).sum()
    n_classes = len(torch.unique(true_y))
    assert preds.shape == (len(true_y), n_classes)
    mask = torch.empty(
        preds.shape, dtype=torch.bool, device=device).fill_(False)
    mask[torch.arange(len(true_y), device=device), true_y] = True
    penalty = torch.sum(F.relu(1.-preds.masked_select(mask))**2) +\
                torch.sum(F.relu(1.+preds.masked_select(~mask))**2)
    loss = loss + C*penalty
    return loss