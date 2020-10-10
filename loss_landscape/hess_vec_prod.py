import torch
import time
import numpy as np
from torch import nn
from torch.autograd import Variable
from scipy.sparse.linalg import LinearOperator, eigsh
from tqdm import tqdm
import sys

sys.path.append('..')
from algorithm_trainer.algorithm_trainer import get_labels

################################################################################
#                              Supporting Functions
################################################################################
def npvec_to_tensorlist(vec, params):
    """ Convert a numpy vector to a list of tensor with the same dimensions as params

        Args:
            vec: a 1D numpy vector
            params: a list of parameters from net

        Returns:
            rval: a list of tensors with the same shape as params
    """
    loc = 0
    rval = []
    for p in params:
        numel = p.data.numel()
        rval.append(torch.from_numpy(vec[loc:loc+numel]).view(p.data.shape).float())
        loc += numel
    assert loc == vec.size, 'The vector has more elements than the net has parameters'
    return rval


def gradtensor_to_npvec(net, include_bn=False):
    """ Extract gradients from net, and return a concatenated numpy vector.

        Args:
            net: trained model
            include_bn: If include_bn, then gradients w.r.t. BN parameters and bias
            values are also included. Otherwise only gradients with dim > 1 are considered.

        Returns:
            a concatenated numpy vector containing all gradients
    """
    filter = lambda p: include_bn or len(p.data.size()) > 1
    return np.concatenate([p.grad.data.cpu().numpy().ravel() for p in net.parameters() if filter(p)])


################################################################################
#                  For computing Hessian-vector products
################################################################################
def eval_hess_vec_prod(vec, params, net, algorithm, dataloader, datamgr, criterion, include_bn=False, use_cuda=False):
    """
    Evaluate product of the Hessian of the loss function with a direction vector "vec".
    The product result is saved in the grad of net.

    Args:
        vec: a list of tensor with the same dimensions as "params".
        params: the parameter list of the net (ignoring biases and BN parameters).
        net: model with trained parameters.
        criterion: loss function.
        dataloader: dataloader for the dataset.
        use_cuda: use GPU.
    """


    net.cuda()
    vec = [v.cuda() for v in vec]
    net.eval()
    net.zero_grad() # clears grad for every parameter in the net
    # meta-learning task configurations
    n_way = datamgr.n_way
    n_shot = datamgr.n_shot
    n_query = datamgr.n_query
    batch_sz = datamgr.batch_size
    print(f"n_way: {n_way}, n_shot: {n_shot}, n_query: {n_query}, batch_sz: {batch_sz}")
    if isinstance(algorithm._model, torch.nn.DataParallel):
        obj = algorithm._model.module
    else:
        obj = algorithm._model
    if hasattr(obj, 'fc') and hasattr(obj.fc, 'scale_factor'):
        scale = obj.fc.scale_factor
    elif hasattr(obj, 'scale_factor'):
        scale = obj.scale_factor
    else:
        scale = 10.    
    
    print("Setting scale to ", scale)
    # iterator
    iterator = tqdm(enumerate(dataloader, start=1),
            leave=False, file=sys.stdout, initial=1, position=0)
    avg_loss = 0. 
    n_batches = 0
    print("iterator", len(dataloader))
    for i, batch in iterator:
        ############## covariates #############
        x_batch, y_batch = batch
        # work around for the reduced dataset in plot_eigen
        x_batch = x_batch[0]
        y_batch = y_batch[0]
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
        # forward pass on updated model
        logits, measurements_trajectory = algorithm.inner_loop_adapt(
            query=query_x, support=shots_x, 
            support_labels=shots_y, scale=scale)
        assert len(set(shots_y)) == len(set(query_y))
        logits = scale * logits.reshape(-1, logits.size(-1))
        query_y = query_y.reshape(-1)
        assert logits.size(0) == query_y.size(0)
        test_loss_after_adapt = criterion(logits, query_y)
        grad_f = torch.autograd.grad(test_loss_after_adapt, inputs=params, create_graph=True, retain_graph=True)
        # Compute inner product of gradient with the direction vector
        prod = Variable(torch.zeros(1)).type(type(grad_f[0].data))
        for (g, v) in zip(grad_f, vec):
            prod = prod + (g * v).cpu().sum()
        prod.backward()
        n_batches += 1
        avg_loss += test_loss_after_adapt.item()

    filter = lambda p: include_bn or len(p.data.size()) > 1
    norm_Hv = 0.
    for p in net.parameters():
        if filter(p):
            p.grad = p.grad/n_batches
            norm_Hv += (p.grad * p.grad).sum().item()

    print(f"avg_loss : {avg_loss / len(dataloader)}")
    print(f"||Hv||_2 : {np.sqrt(norm_Hv)}")

################################################################################
#                  For computing Eigenvalues of Hessian
################################################################################
def min_max_hessian_eigs(net, algorithm, dataloader, datamgr, criterion, rank=0, use_cuda=False, verbose=True):
    """
        Compute the largest and the smallest eigenvalues of the Hessian marix.

        Args:
            net: the trained model.
            dataloader: dataloader for the dataset, may use a subset of it.
            criterion: loss function.
            rank: rank of the working node.
            use_cuda: use GPU
            verbose: print more information

        Returns:
            maxeig: max eigenvalue
            mineig: min eigenvalue
            hess_vec_prod.count: number of iterations for calculating max and min eigenvalues
    """

    params = [p for p in net.parameters() if len(p.size()) > 1]
    N = sum(p.numel() for p in params)

    def hess_vec_prod(vec):
        hess_vec_prod.count += 1  # simulates a static variable
        vec = npvec_to_tensorlist(vec, params)
        start_time = time.time()
        eval_hess_vec_prod(vec, params, net, algorithm, dataloader, datamgr, criterion, use_cuda)
        prod_time = time.time() - start_time
        if verbose and rank == 0: print("   Iter: %d  time: %f" % (hess_vec_prod.count, prod_time))
        return gradtensor_to_npvec(net)
        
    hess_vec_prod.count = 0
    if verbose and rank == 0: print("Rank %d: computing max eigenvalue" % rank)

    A = LinearOperator((N, N), matvec=hess_vec_prod)
    eigvals, eigvecs = eigsh(A, k=1, tol=1e-2)
    maxeig = eigvals[0]
    if verbose and rank == 0: print('max eigenvalue = %f' % maxeig)

    # If the largest eigenvalue is positive, shift matrix so that any negative eigenvalue is now the largest
    # We assume the smallest eigenvalue is zero or less, and so this shift is more than what we need
    shift = maxeig*.51
    def shifted_hess_vec_prod(vec):
        return hess_vec_prod(vec) - shift*vec

    if verbose and rank == 0: print("Rank %d: Computing shifted eigenvalue" % rank)

    A = LinearOperator((N, N), matvec=shifted_hess_vec_prod)
    eigvals, eigvecs = eigsh(A, k=1, tol=1e-2)
    eigvals = eigvals + shift
    mineig = eigvals[0]
    if verbose and rank == 0: print('min eigenvalue = ' + str(mineig))

    if maxeig <= 0 and mineig > 0:
        maxeig, mineig = mineig, maxeig

    return maxeig, mineig, hess_vec_prod.count
