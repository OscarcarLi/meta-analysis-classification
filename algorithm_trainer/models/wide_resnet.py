import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from  algorithm_trainer.models.dropblock import DropBlock
import torch
import sys, os
import numpy as np
import random
import math
from torch.nn.utils.weight_norm import WeightNorm
from collections import defaultdict



class distLinear(nn.Module):
    def __init__(self, indim, outdim):
        super(distLinear, self).__init__()
        self.L = nn.Linear(indim, outdim, bias = False)
        self.class_wise_learnable_norm = True  #See the issue#4&8 in the github 
        if self.class_wise_learnable_norm:      
            WeightNorm.apply(self.L, 'weight', dim=0) #split the weight update component to direction and norm      

        if outdim <=200:
            self.scale_factor = 2; #a fixed scale factor to scale the output of cos value into a reasonably large input for softmax, for to reproduce the result of CUB with ResNet10, use 4. see the issue#31 in the github 
        else:
            self.scale_factor = 10; #in omniglot, a larger scale factor is required to handle >1000 output classes.

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim =1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm+ 0.00001)
        if not self.class_wise_learnable_norm:
            L_norm = torch.norm(self.L.weight.data, p=2, dim =1).unsqueeze(1).expand_as(self.L.weight.data)
            self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
        cos_dist = self.L(x_normalized) #matrix product by forward function, but when using WeightNorm, this also multiply the cosine distance by a class-wise learnable norm, see the issue#4&8 in the github
        scores = self.scale_factor* (cos_dist) 

        return scores




class avgLinear(nn.Module):

    def __init__(self, indim, outdim, lambd=0.5, gamma=0.99):
        super(avgLinear, self).__init__()
        # self.L = nn.Parameter(torch.randn(outdim, indim), requires_grad=True) 
        self.L = None
        self.Lg = nn.Linear(indim, outdim, bias = False)
        self.gbeta = nn.Parameter(torch.FloatTensor([1.0]))
        self.scale_factor = nn.Parameter(torch.FloatTensor([1.0]))
        # self.scale_factor = 2.0
        # self.M = torch.nn.Linear(indim, indim, bias=False)
        # self.M = nn.Parameter(torch.eye(indim), requires_grad=False)
        self.lambd = lambd
        self.gamma = gamma
        self.indim = indim
        self.outdim = outdim
        self.class_count = defaultdict(int)

        # for param in self.Lg.parameters():
        #     param.requires_grad = False

    def K(self, a, b):
        """ linear kernel
        """
        # return self.M(a) @ self.M(b).T 
        return a @ b.T

    def forward(self, x):
        # Lg_norm = torch.norm(self.Lg.weight.data, p=2, dim =1).unsqueeze(1).expand_as(self.Lg.weight.data)
        # self.Lg.weight.data = self.Lg.weight.data.div(Lg_norm + 0.00001)
        # scores = self.scale_factor * (self.gbeta * (self.K(x, self.L)) + self.K(x, self.Lg.weight))
        # scores = self.scale_factor * ((self.K(x, self.L) - self.K(x, self.L.detach()))  + self.K(x, self.Lg.weight))
        scores = (self.scale_factor ** 2) * (self.K(x, self.L))
        return scores

    def update_L(self, x, y):

        c_mat = []
        for c in np.arange(self.outdim):
            # np.unique(y.cpu().numpy()):
            if len(x[y==c, :]) > 0:
                c_feat = torch.mean(x[y==c, :], dim=0)
                c_mat.append(c_feat)
            else:
                c_mat.append(self.Lg.weight[c, :])
            # c_feat = c_feat.div(torch.norm(c_feat, p=2)+ 0.00001)
            # self.L[c, :] = (1 - self.momentum) * c_feat + self.momentum * self.Lg.weight[c, :]
        # print(len(c_mat)) 
        # print(c_mat[0].shape)
        c_mat = torch.stack(c_mat, dim=0)
        # print(c_mat.shape)
        # self.L = self.lambd * self.Lg.weight + c_mat
        self.L = self.Lg.weight.div(torch.max(torch.norm(self.Lg.weight, dim=1))) + self.lambd * c_mat



    def update_Lg(self, x, y):

        # print("recvd:", "x:", x.shape, "y:", y.shape)
        with torch.no_grad():
            # self.Lg.weight = self.L.detach()

            for c in np.unique(y.cpu().numpy()):
                c_feat = torch.mean(x[y==c, :], dim=0)
                # c_feat = c_feat.div(torch.norm(c_feat, p=2)+ 0.00001)
                self.Lg.weight[c, :] = self.gamma * self.Lg.weight[c, :] + (1. - self.gamma) * c_feat


    def update_Lg_full(self, x, y):

        # print("recvd:", "x:", x.shape, "y:", y.shape)
        with torch.no_grad():
            for c in np.unique(y.cpu().numpy()):
                c_feat = torch.sum(x[y==c, :], dim=0)
                self.Lg.weight[c, :] = self.Lg.weight[c, :] + c_feat
                self.class_count[c] += x[y==c, :].shape[0]


    def divide_Lg(self):

        for c in self.class_count:
            self.Lg.weight[c, :] = self.Lg.weight[c, :].div(self.class_count[c])
            self.class_count[c] = 0
        
        


    def project_Lg(self):

        Lg_norm = torch.norm(self.Lg.weight.data, p=2, dim =1).unsqueeze(1).expand_as(self.Lg.weight.data)
        self.Lg.weight.data = self.Lg.weight.data.div(Lg_norm + 0.00001)
        

    def compute_loss(self):   

        # I = torch.eye(self.outdim, device=self.L.device)
        # print("L:", self.L)
        # print("Lg:", self.Lg.weight.t())
        
        # print(self.L @ self.Lg.weight.t())
        # print(torch.sum((self.L @ self.Lg.weight.t())**2))

        # L_n = torch.norm(self.L, dim=1, p=2)
        # Lg_n = torch.norm(self.Lg.weight, dim=1, p=2)

        # loss = torch.sum((L_n * Lg_n - torch.diag(self.L @ self.Lg.weight.t()))**2) 
        loss = torch.sum((self.L - self.Lg.weight)**2)
        return loss







class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

    
    
class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)


def to_one_hot(inp,num_classes):

    y_onehot = torch.FloatTensor(inp.size(0), num_classes)
    if torch.cuda.is_available():
        y_onehot = y_onehot.cuda()

    y_onehot.zero_()
    x = inp.type(torch.LongTensor)
    if torch.cuda.is_available():
        x = x.cuda()

    x = torch.unsqueeze(x , 1)
    y_onehot.scatter_(1, x , 1)
    
    return Variable(y_onehot,requires_grad=False)
    # return y_onehot


def mixup_data(x, y, lam):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
   
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    if torch.cuda.is_available():
        index = index.cuda()
    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam

    
class WideResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=10, stride=1,
            num_classes=200, classifier_type='distance-classifier', no_fc_layer=False, add_bias=False):

        dropRate = 0.5
        flatten = True
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, stride, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and linear
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.nChannels = nChannels[3]
        

        # classifier creation
        self.final_feat_dim = nChannels[3]
        self.classifier_type  = classifier_type
        self.num_classes = num_classes
        
        if no_fc_layer is True:
            self.fc = None
        elif classifier_type == 'linear':
            self.fc = nn.Linear(self.final_feat_dim, num_classes)
            self.fc.bias.data.fill_(0)
        elif classifier_type == 'distance-classifier':
            self.fc = distLinear(self.final_feat_dim, num_classes)
        elif classifier_type == 'avg-classifier':
            self.fc = avgLinear(self.final_feat_dim, num_classes)
        else:
            raise ValueError("classifier type not found")

        self.no_fc_layer = no_fc_layer
        self.add_bias = add_bias
        self.scale = nn.Parameter(torch.FloatTensor([1.0]))


        # initialization        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    
    def forward(self, x, target=None, mixup=False, 
        mixup_hidden=True, mixup_alpha=None, lam = 0.4, features_only=False):

        if target is not None: 
            if mixup_hidden:
                layer_mix = random.randint(0,3)
            elif mixup:
                layer_mix = 0
            else:
                layer_mix = None   

            out = x

            target_a = target_b  = target

            if layer_mix == 0:
                out, target_a , target_b , lam = mixup_data(out, target, lam=lam)

            out = self.conv1(out)
            out = self.block1(out)


            if layer_mix == 1:
                out, target_a , target_b , lam  = mixup_data(out, target, lam=lam)

            out = self.block2(out)

            if layer_mix == 2:
                out, target_a , target_b , lam = mixup_data(out, target, lam=lam)


            out = self.block3(out)
            if  layer_mix == 3:
                out, target_a , target_b , lam = mixup_data(out, target, lam=lam)

            out = self.relu(self.bn1(out))
            out = F.avg_pool2d(out, out.size()[2:])
            out = out.view(out.size(0), -1)

            out_norm = torch.norm(out, dim=1, keepdim=True)+0.00001
            out = out.div(out_norm)
            if features_only:
                return out
            if self.add_bias and self.fc is None:
                out = torch.cat([out, 10.*torch.ones((out.size(0), 1), device=out.device)], dim=-1)
            elif self.fc is not None:
                out = self.fc.forward(out)
        
            return out, target_a, target_b

        else: 
            out = x
            out = self.conv1(out)
            out = self.block1(out)
            out = self.block2(out)
            out = self.block3(out)
            out = self.relu(self.bn1(out))
            out = F.avg_pool2d(out, out.size()[2:])
            out = out.view(out.size(0), -1)

            out_norm = torch.norm(out, dim=1, keepdim=True)+0.00001
            out = out.div(out_norm)
            if features_only:
                return out
            if self.add_bias and self.fc is None:
                out = torch.cat([out, 10.*torch.ones((out.size(0), 1), device=out.device)], dim=-1)
            elif self.fc is not None:
                out = self.fc.forward(out)
        
            return out
                  
        
def wrn28_10(**kwargs):
    model = WideResNet(depth=28, widen_factor=10, stride=1, **kwargs)
    return model