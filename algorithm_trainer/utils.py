import subprocess

import torch
import torch.nn.functional as F


def set_lr(lr, epoch, cycle, start_lr, end_lr):
    if epoch % cycle == 0:
        return start_lr
    return lr + (end_lr - start_lr) / cycle  


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



def add_fc(old_model, prefc_feature_sz, num_classes):
    if isinstance(old_model, torch.nn.DataParallel):
        old_model.module.fc = torch.nn.Linear(
            prefc_feature_sz, num_classes).cuda()
        new_model = torch.nn.DataParallel(
            old_model.module, device_ids=range(torch.cuda.device_count()))
    else:
        old_model.fc = torch.nn.Linear(
            prefc_feature_sz, num_classes).cuda()
        new_model = old_model
    return new_model


def remove_fc(model):
    if isinstance(model, torch.nn.DataParallel):
        model.module.fc = None
    else:
        model.fc = None
    return model





def get_swa_model(swa_model, model, alpha=1):
    for param1, param2 in zip(swa_model.parameters(), model.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha
    return swa_model


def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def bn_update(loader, model):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.
        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    for input, _ in loader:
        input = input.cuda()
        input_var = torch.autograd.Variable(input)
        b = input_var.data.size(0)

        momentum = b / (n + b)
        for module in momenta.keys():
            module.momentum = momentum

        model(input_var)
        n += b

    model.apply(lambda module: _set_momenta(module, momenta))