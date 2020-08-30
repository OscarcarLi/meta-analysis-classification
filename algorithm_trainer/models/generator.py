import torch.nn as nn
from collections import defaultdict
import torch
import numpy as np

class Generator(nn.Module):
    
    def __init__(self, indim, outdim, device='cuda'):
        super(Generator, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.eps = nn.Parameter(torch.zeros(outdim, indim), requires_grad=True)
        self.means = nn.Parameter(torch.zeros(outdim, indim), requires_grad=False)
        self.stddev = nn.Parameter(torch.zeros(outdim, indim), requires_grad=False)
        self.class_count = defaultdict(int)
        self.device = device
        

    def update_mean(self, x, y):

        with torch.no_grad():
            for c in np.unique(y.cpu().numpy()):
                c_feat = torch.sum(x[y==c, :], dim=0)
                self.means[c, :] = self.means[c, :] + c_feat
                self.class_count[c] += x[y==c, :].shape[0]


    def update_stddev(self, x, y):

        with torch.no_grad():
            for c in np.unique(y.cpu().numpy()):
                self.stddev[c, :] = self.stddev[c, :] + torch.sum((x[y==c, :] - self.means[c, :])**2, dim=0)


    def div_mean(self):

        for c in np.arange(self.outdim):
            self.means[c] /= self.class_count[c]


    def div_stddev(self):
        
        for c in np.arange(self.outdim):
            self.stddev[c] /= self.class_count[c]
            self.stddev[c] = torch.sqrt(self.stddev[c])
            


    def reset(self):

        # self.eps.fill_(0.)
        self.means.fill_(0.)
        self.stddev.fill_(0.)
        self.class_count = defaultdict(int)
        

    def generate(self):
        return self.means + (torch.clamp(self.eps, -2.0, 2.0) * self.stddev)
