import cvxpy as cp 
from cvxpylayers.torch import CvxpyLayer 
import torch  
from sklearn.linear_model import LogisticRegression
import numpy as np

# torch stuff
x = torch.randn(5, 1000, requires_grad=True) 
A = torch.randn(1000, 16, requires_grad=True)
y = torch.arange(5).numpy()
modulated_x = torch.cat([x @ A, torch.ones(x.shape[0], 1)], dim=-1) 
n_classes = len(y)
n_samples = x.shape[0]

# cvxpy variables
C = 0.01
W = cp.Variable((5, 17)) 
xi = cp.Variable((5, 5))
x_cp = cp.Parameter(modulated_x.shape)
constraints = [xi >= 0]
z = x_cp @ W.T
for i in range(n_samples):
    for j in range(n_classes):
        if y[i] != j:
            constraints.append(z[i][y[i]] >= z[i][j] - xi[i][j])
obj = cp.Minimize(0.5 * cp.sum_squares(W) + C* cp.sum(xi))
problem = cp.Problem(obj, constraints) 

print(problem.is_dpp())
print(problem.is_dpp())



def test_logistic():

    np.random.seed(42)
    c = 5
    m = 5
    n = 1600
    beta = cp.Variable((c, n))
    
    X = np.random.random((m, n))
    Y = np.arange(5)
        
    # print((X @ beta.T).shape)
    cY = np.zeros((5,5))
    cY[np.arange(5),y] = 1. 
    # print((cp.multiply(cY, X @ beta.T)).shape)
    log_likelihood = cp.sum(cp.multiply(cY, X @ beta.T)) -cp.sum(cp.log_sum_exp(
        X @ beta.T, axis=1))
    
    # for y in set(Y):
    #     cY = (Y==y).astype(np.float)
    #     log_likelihood += cp.sum(
    #         cp.multiply(cY, X @ beta[y, :])
    #     )
    
    problem = cp.Problem(cp.Maximize(log_likelihood))
    problem.solve(verbose=False)
    # print(beta.value)
    # print(np.dot(X, beta.T.value))
    # print(np.argmax(np.dot(X, beta.T.value), 1))

def test_logistic_sklearn():

    np.random.seed(42)
    c = 5
    m = 5
    n = 1600
    beta = cp.Variable((c, n))
    
    X = np.random.random((m, n))
    Y = np.arange(5)
    lr_model = LogisticRegression(solver='lbfgs', penalty='l2', 
                C=100000, # now use _l2_lambda instead of 2 * _l2_lambda
                tol=1e-6, max_iter=150,
                multi_class='multinomial', fit_intercept=False)
    lr_model.fit(X, y)



if __name__=='__main__':
    test_logistic()