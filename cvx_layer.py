import cvxpy as cp 
from cvxpylayers.torch import CvxpyLayer 
import torch  

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


