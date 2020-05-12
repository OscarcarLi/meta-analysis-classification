import numpy as np
from scipy.special import softmax

def logistic_regression_grad_with_respect_to_w(X, y, w):
    '''
    X, y, w numpy arrays
       shape
    X  N, (d+1) last dimension the bias
    y  N integer class identity
    w  C, (d+1) number of classes C

    return grad C*(d+1), 1
    '''
    preds = np.matmul(X, w.T)
    p = softmax(preds, axis=1) # the probability matrix N, C

    C = w.shape[0]
    N = X.shape[0]

    result = np.zeros((C * X.shape[1], 1))
    I = np.eye(C, dtype=np.float32)
    for i in range(X.shape[0]):
        result += np.kron(p[i].reshape(-1, 1) - I[:, y[i]: y[i]+1], X[i].reshape(-1, 1))
    result /= N
    
    return result


def logistic_regression_hessian_pieces_with_respect_to_w(X, y, w):
    '''
    X, y, w numpy arrays
       shape
    X  N, (d+1) last dimension the bias
    y  N integer class identity
    w  C, (d+1) number of classes C
    
    return hessian components:
    diag N(C+1) diag has been divided by N for each component
    X    N(C+1), C(d+1)
    the final hessian can be computed as X.T @ np.diag(diag) @ X
    '''
    preds = np.matmul(X, w.T)
    p = softmax(preds, axis=1) # the probability matrix N, C

    C = w.shape[0]
    N = X.shape[0]
    d = X.shape[1] - 1
    
    Xbar = np.zeros(shape=(N * (C+1), C * (d+1)))
    diag = []
    
    for i in range(N):
        for j in range(C):
            Xbar[i * C + j, j * (d+1): (j+1) * (d+1)] = X[i]        
        diag.extend(p[i] / N)
    for i in range(N):
        Xbar[N * C + i, :] = np.kron(p[i], X[i])
        diag.append(-1 / N)

    return diag, Xbar


def logistic_regression_hessian_with_respect_to_w(X, y, w):
    '''
    X, y, w numpy arrays
       shape
    X  N, (d+1) last dimension the bias
    y  N integer class identity
    w  C, (d+1) number of classes C
    
    return hessian matrix C*(d+1), C*(d+1)
    '''
    diag, Xbar = logistic_regression_hessian_pieces_with_respect_to_w(X, y, w)
    result = np.matmul(np.matmul(Xbar.T, np.diag(diag)), Xbar)

    return result

def logistic_regression_hessian_old_with_respect_to_w(X, y, w):
    # currently not used
    '''
    X, y, w numpy arrays
       shape
    X  N, (d+1) last dimension the bias
    y  N integer class identity
    w  C, (d+1) number of classes C
    
    return hessian matrix C*(d+1), C*(d+1)
    '''
    preds = np.matmul(X, w.T)
    p = softmax(preds, axis=1) # the probability matrix N, C
    
    C = w.shape[0]
    N = X.shape[0]
    
    result = np.zeros((C * X.shape[1], C * X.shape[1]), dtype=np.float32)
    for i in range(N):
        result += np.kron(np.diag(p[i]) - np.outer(p[i], p[i]), np.outer(X[i], X[i]))
    result /= N

    return result


def logistic_regression_mixed_derivatives_with_respect_to_w_then_to_X(X, y, w):
    '''
    X, y, w numpy arrays
       shape
    X  N, (d+1) last dimension the bias
    y  N integer class identity
    w  C, (d+1) number of classes C

    return mixed partial matrix C*(d+1), N*(d+1)
    '''
    preds = np.matmul(X, w.T)
    p = softmax(preds, axis=1) # the probability matrix N, C
    
    C = w.shape[0]
    N = X.shape[0]
    
    d = X.shape[1] - 1
    
    I_C = np.eye(C, dtype=np.float32)
    I_dp1 = np.eye(d + 1, dtype=np.float32)

    result = np.kron((p - I_C[y]).T, I_dp1)
    
    for i in range(N):
        weighted_w = np.matmul(np.diag(p[i]), np.matmul(I_C - p[i], w)) # broadcasting
        for j in range(C):
            result[j*(d+1):(j+1)*(d+1), i*(d+1): (i+1)*(d+1)] += np.outer(X[i], weighted_w[j])
    result /= N

    return result