import torch.nn as nn
import torch
import torch.nn.functional as F
from algorithm_trainer.models.dropblock import DropBlock
from torch.autograd import Variable
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm
from collections import defaultdict



class CvxClasifier(nn.Module):

    def __init__(self, indim, outdim, lambd_start=0.8, lambd_end=0.8, metric='cosine', projection=True):
        super(CvxClasifier, self).__init__()
        self.L = None
        self.Lg = nn.Linear(indim, outdim, bias = False)
        self.scale_factor = nn.Parameter(torch.FloatTensor([10.0]))
        self.indim = indim
        self.outdim = outdim
        self.class_count = defaultdict(int)
        self.lambd_start = lambd_start
        self.lambd_end = lambd_end
        self.lambd = lambd_start
        self.metric = metric
        self.projection = projection
        print("Classifier metric is ", self. metric)

    def update_lambd(self):
        if (self.lambd < self.lambd_end and self.lambd >= self.lambd_start) or (self.lambd > self.lambd_end and self.lambd <= self.lambd_start): 
            self.lambd = self.lambd + (self.lambd_end - self.lambd_start) / self.n_epochs
        print(f"Current avg classifier lambd {self.lambd}")

    def K(self, a, b):
        """ linear kernel
        """
        if self.metric == 'cosine':
            return a @ b.T
        elif self.metric == 'euclidean':
            n_way = b.size(0)
            d = b.size(1)
            AB = a @ b.T
            # total_n_query x n_way
            AA = (a * a).sum(dim=1, keepdim=True)
            # total_n_query x 1
            BB = (b * b).sum(dim=1, keepdim=True).reshape(1, n_way)
            # 1 x n_way
            logits_query = AA.expand_as(AB) - 2 * AB + BB.expand_as(AB)
            # euclidean distance 
            logits_query = -logits_query
            # batch_size x total_n_query x n_way
            logits_query = logits_query / d
            # normalize
            return logits_query

    def forward(self, x):
        scores = torch.abs(self.scale_factor) * (self.K(x, self.L))
        return scores

    def update_L(self, x, y):

        if self.projection and self.lambd > 0.:
            self.Lg.weight.div(torch.max(torch.norm(self.Lg.weight, dim=1)))
        if self.lambd == 1.:
            self.L = self.Lg.weight
        else:
            c_mat = []
            for c in np.arange(self.outdim):
                c_feat = torch.mean(x[y==c, :], dim=0)
                c_mat.append(c_feat)
            c_mat = torch.stack(c_mat, dim=0)
            if self.lambd == 0.:
                self.L = c_mat
            else:
                self.L = self.lambd * self.Lg.weight + (1. - self.lambd) * c_mat



class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, retain_activation=True, activation='ReLU'):
        super(ConvBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        if retain_activation:
            if activation == 'ReLU':
                self.block.add_module("ReLU", nn.ReLU(inplace=True))
            elif activation == 'LeakyReLU':
                self.block.add_module("LeakyReLU", nn.LeakyReLU(0.1))
            elif activation == 'Softplus':
                self.block.add_module("Softplus", nn.Softplus())
        self.block.add_module("MaxPool2d", nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        
    def forward(self, x):
        out = self.block(x)
        return out

# Embedding network used in Matching Networks (Vinyals et al., NIPS 2016), Meta-LSTM (Ravi & Larochelle, ICLR 2017),
# MAML (w/ h_dim=z_dim=32) (Finn et al., ICML 2017), Prototypical Networks (Snell et al. NIPS 2017).



def conv_block(in_channels, out_channels):
    bn = nn.BatchNorm2d(out_channels)
    nn.init.uniform_(bn.weight) # for pytorch 1.2 or later
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        bn,
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class Convnet(nn.Module):

    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )
        self.out_channels = 1600



class Conv64(nn.Module):
    def __init__(self, num_classes, classifier_type='avg-linear', x_dim=3, h_dim=64, z_dim=64, 
        retain_last_activation=True, activation='ReLU', add_bias=False, projection=False, classifier_metric='euclidean', lambd=0.):
        super(Conv64, self).__init__()
        
        self.encoder = nn.Sequential(
            conv_block(x_dim, h_dim),
            conv_block(h_dim, h_dim),
            conv_block(h_dim, h_dim),
            conv_block(h_dim, z_dim),
        )
        
        # self.encoder = nn.Sequential(
        #   ConvBlock(x_dim, h_dim, activation=activation),
        #   ConvBlock(h_dim, h_dim, activation=activation),
        #   ConvBlock(h_dim, h_dim, activation=activation),
        #   ConvBlock(h_dim, z_dim, retain_activation=retain_last_activation, activation=activation),
        # )
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

        # classifier creation
        self.projection = projection
        print("Unit norm projection is ", self.projection)
        print("Avg pool is always False for Conv64")
        self.final_feat_dim = 1600
        self.num_classes = num_classes
        self.no_fc_layer = (classifier_type == "no-classifier")
        self.classifier_type = classifier_type

        if self.no_fc_layer is True:
            self.fc = None
            self.scale_factor = nn.Parameter(torch.FloatTensor(1).fill_(10.0), requires_grad=True)
        elif self.classifier_type == 'linear':
            self.fc = nn.Linear(self.final_feat_dim, num_classes)
            self.fc.bias.data.fill_(0)
        elif self.classifier_type == 'cvx-classifier':
            self.fc = CvxClasifier(self.final_feat_dim, num_classes, 
            metric=classifier_metric, lambd_start=lambd, lambd_end=lambd, projection=self.projection)
        else:
            raise ValueError("classifier type not found")

        self.add_bias = add_bias
        # self.scale_factor = nn.Parameter(torch.FloatTensor([10.0]))
        # self.scale_factor = 1.
        

    def forward(self, x, features_only=True):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        
        # hypersphere projection
        if self.projection:
            x_norm = torch.norm(x, dim=1, keepdim=True)+0.00001
            x = x.div(x_norm)

        if features_only:
            return x
        if self.add_bias and self.fc is None:
            x = torch.cat([x, 10.*torch.ones((x.size(0), 1), device=x.device)], dim=-1)
        elif self.fc is not None:
            x = self.fc.forward(x)
        
        return x
