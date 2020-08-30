# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from torch.autograd import Variable
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm
from  algorithm_trainer.models.dropblock import DropBlock

# Basic ResNet model

def init_layer(L):
    # Initialization using fan-in
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0]*L.kernel_size[1]*L.out_channels
        L.weight.data.normal_(0,math.sqrt(2.0/float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)

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
    def __init__(self, indim, outdim, momentum=0.99):
        super(avgLinear, self).__init__()
        self.L = nn.Parameter(torch.randn(outdim, indim), requires_grad=False)
        self.Lg = nn.Linear(indim, outdim, bias = False)
        self.gbeta = nn.Parameter(torch.FloatTensor([1.0]))
        self.scale_factor = nn.Parameter(torch.FloatTensor([1.0]))
        self.M = torch.nn.Linear(indim, indim, bias=False)
        self.momentum = momentum
        self.indim = indim
        self.outdim = outdim

        for param in self.Lg.parameters():
            param.requires_grad = False
        
        
        # if outdim <=200:
        #     self.scale_factor = 2; #a fixed scale factor to scale the output of cos value into a reasonably large input for softmax, for to reproduce the result of CUB with ResNet10, use 4. see the issue#31 in the github 
        # else:
        #     self.scale_factor = 10; #in omniglot, a larger scale factor is required to handle >1000 output classes.

    def K(a, b):
        """ linear kernel
        """
        return self.M(a) @ self.M(b).T 

    def forward(self, x):
        # x_norm = torch.norm(x, p=2, dim =1).unsqueeze(1).expand_as(x)
        # x_normalized = x.div(x_norm+ 0.00001)
        Lg_norm = torch.norm(self.Lg.weight.data, p=2, dim =1).unsqueeze(1).expand_as(self.Lg.weight.data)
        self.Lg.weight.data = self.Lg.weight.data.div(Lg_norm + 0.00001)
        # self.L = self.L.to(x.device)
        # (x @ self.L.T) +
        scores = self.scale_factor * (self.gbeta * (self.K(x, self.L)) + self.K(x, self.Lg.weight))
        #matrix product by forward function, but when using WeightNorm, this also multiply the cosine distance by a class-wise learnable norm, see the issue#4&8 in the github
        # scores = self.scale_factor* (cos_dist) 
        return scores

    def update(self, x, y):

        # with torch.no_grad():
            # x_norm = torch.norm(x, p=2, dim =1).unsqueeze(1).expand_as(x)
            # x = x.div(x_norm+ 0.00001)
        # self.L = torch.randn(self.outdim, self.indim, device=x.device)
        for c in np.unique(y.cpu().numpy()):
            c_feat = torch.mean(x[y==c, :], dim=0)
            # c_feat = c_feat.div(torch.norm(c_feat, p=2)+ 0.00001)
            # self.L.weight[c, :] = self.L.weight[c, :] + c_feat
            # self.L.weight[c, :] = self.L.weight[c, :].div(len(self.L.weight[c, :]))
            # self.L.weight[c, :] = self.momentum * self.L.weight[c, :] + (1-self.momentum) * c_feat
            c_feat = c_feat.div(torch.norm(c_feat, p=2)+ 0.00001)
            self.L.data[c, :] = c_feat
        # assert self.L.shape[0] == self.outdim
        # self.L = c_feat
        # L_norm = torch.norm(self.L.weight.data, p=2, dim =1).unsqueeze(1).expand_as(self.L.weight.data)
        # self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
        
                

    def update_full(self, x, y):

        with torch.no_grad():
            # x_norm = torch.norm(x, p=2, dim =1).unsqueeze(1).expand_as(x)
            # x = x.div(x_norm+ 0.00001)
            for c in np.unique(y.cpu().numpy()):
                c_feat = torch.sum(x[y==c, :], dim=0) / 600.
                # c_feat = c_feat.div(torch.norm(c_feat, p=2)+ 0.00001)
                # self.L.weight[c, :] = self.L.weight[c, :] + c_feat
                # self.L.weight[c, :] = self.L.weight[c, :].div(len(self.L.weight[c, :]))
                self.Lg.weight[c, :] = self.Lg.weight[c, :] + c_feat
            # L_norm = torch.norm(self.L.weight.data, p=2, dim =1).unsqueeze(1).expand_as(self.L.weight.data)
            # self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
        
                


class gaussianDA(nn.Module):
    
    def __init__(self, indim, outdim):
        super(gaussianDA, self).__init__()
        self.mu = nn.Linear(indim, outdim, bias=False)
        self.inv_diag_C = nn.Linear(indim, outdim, bias=False) 
        self.indim = indim
        self.outdim = outdim
        

    def forward(self, x):
        
        batch_sz, feature_sz = x.size()
        assert feature_sz == self.indim
        x = x.unsqueeze(1).repeat_interleave(self.outdim, dim=1).unsqueeze(2) 
        # batch_sz x outdim x 1 x feature_sz
        mu = self.mu.weight.unsqueeze(0).repeat_interleave(batch_sz, dim=0).unsqueeze(2)
        # batch_sz x outdim x 1 x feature_sz
        inv_diag_C = torch.diag_embed(self.inv_diag_C.weight).unsqueeze(0).repeat_interleave(batch_sz, dim=0)
        # batch_sz x outdim x feature_sz x feature_sz

        # reshape for bmm
        x = x.reshape(-1, 1, self.indim)
        # (batch_sz*outdim) x 1 x feature_sz
        mu = mu.reshape(-1, 1, self.indim)
        # (batch_sz*outdim) x 1 x feature_sz
        inv_diag_C = inv_diag_C.reshape(-1, self.indim, self.indim)
        # (batch_sz*outdim) x feature_sz x feature_sz
        
        # print(f"x: {x.shape}, mu:{mu.shape}, inv_diag_C:{inv_diag_C.shape}")
        # compute logits and reshape
        logits = torch.bmm(torch.bmm(x - mu, inv_diag_C), (x - mu).transpose(1, 2))
        # (batch_sz*outdim) x 1 x 1
        logits = -logits.squeeze()
        # (batch_sz*outdim)
        logits = logits.reshape(batch_sz, self.outdim)
        # batch_sz x outdim
        
        # print("logits", logits.shape)
        return logits



class orthonormalDistLinear(nn.Module):
    
    def __init__(self, indim, outdim, K=5):
        super(orthonormalDistLinear, self).__init__()
        self.K = K
        self.L = torch.nn.Linear(indim, outdim * K, bias=False)
        self.class_scale = torch.nn.Parameter(torch.ones(outdim) ,requires_grad=True)
        # ortho init
        torch.nn.init.orthogonal_(self.L.weight)
        self.indim = indim
        self.outdim = outdim
        

    def forward(self, x):
        
        batch_sz, feature_sz = x.size()
        assert feature_sz == self.indim
        x = x.div(torch.norm(x, dim=1, keepdim=True)+0.00001)
        bare_logits = self.L(x)
        class_logits = bare_logits.reshape(batch_sz, self.outdim, self.K)
        class_logits = class_logits.sum(dim=2)
        class_logits = class_logits * self.class_scale.repeat(batch_sz, 1)
        return class_logits



class prinCompClassifier(nn.Module):
    
    def __init__(self, indim, outdim, momentum=0.9, K=5):
        super(prinCompClassifier, self).__init__()
        self.K = K
        self.running_cov = torch.nn.Parameter(torch.randn(outdim, indim*indim) ,requires_grad=False)
        self.running_pc = torch.nn.Parameter(torch.randn(outdim*K, indim) ,requires_grad=False)
        self.momentum = momentum
        # ortho init
        self.indim = indim
        self.outdim = outdim
        

    def forward(self, x):
        
        batch_sz, feature_sz = x.size()
        assert feature_sz == self.indim
        x = x.div(torch.norm(x, dim=1, keepdim=True)+0.00001)
        proj = x @ self.running_pc.t()
        # batch_sz x (outdim*K)
        proj = (proj.reshape(batch_sz, self.outdim, self.K)) ** 2
        logits = proj.sum(dim=2)
        # batch_sz x outdim
        return logits

    def update_pc(self, x, y):
        """Update running statistics of pc for each class.
        """
        batch_sz, feature_sz = x.size()
        batch_pc = self.running_pc.clone()
        x = x.div(torch.norm(x, dim=1, keepdim=True)+0.00001)
        with torch.no_grad():
            for c in np.unique(y.cpu().numpy()):
                c_feat = x[y==c, :]
                c_feat = c_feat - torch.mean(c_feat, dim=0, keepdim=True)
                c_updated_cov = self.momentum * self.running_cov.data[c, :].reshape(self.indim, self.indim) +\
                                    (1-self.momentum) * (c_feat.t() @ c_feat) * (1. / len(c_feat))
                self.running_cov.data[c, :] = c_updated_cov.reshape(-1)
                pc_update = eigs(
                    self.running_cov.data[c, :].reshape(self.indim, self.indim).cpu().numpy(), self.K)[2].T
                self.running_pc.data[c*self.K:(c+1)*self.K, :] = torch.FloatTensor(
                    np.float32(pc_update), device=c_feat.device) 
                


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        
    def forward(self, x):        
        return x.view(x.size(0), -1)

# Simple Conv Block
class ConvBlock(nn.Module):
    def __init__(self, indim, outdim, pool = True, padding = 1):
        super(ConvBlock, self).__init__()
        self.indim  = indim
        self.outdim = outdim
        self.C      = nn.Conv2d(indim, outdim, 3, padding= padding)
        self.BN     = nn.BatchNorm2d(outdim)
        self.relu   = nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C, self.BN, self.relu]
        if pool:
            self.pool   = nn.MaxPool2d(2)
            self.parametrized_layers.append(self.pool)

        for layer in self.parametrized_layers:
            init_layer(layer)

        self.trunk = nn.Sequential(*self.parametrized_layers)


    def forward(self,x):
        out = self.trunk(x)
        return out

# Simple ResNet Block
class SimpleBlock(nn.Module):
    
    def __init__(self, indim, outdim, half_res):
        super(SimpleBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.C1 = nn.Conv2d(indim, outdim, kernel_size=3, stride=2 if half_res else 1, padding=1, bias=False)
        self.BN1 = nn.BatchNorm2d(outdim)
        self.C2 = nn.Conv2d(outdim, outdim,kernel_size=3, padding=1,bias=False)
        self.BN2 = nn.BatchNorm2d(outdim)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C1, self.C2, self.BN1, self.BN2]

        self.half_res = half_res

        # if the input number of channels is not equal to the output, then need a 1x1 convolution
        if indim!=outdim:
            self.shortcut = nn.Conv2d(indim, outdim, 1, 2 if half_res else 1, bias=False)
            self.BNshortcut = nn.BatchNorm2d(outdim)
            self.parametrized_layers.append(self.shortcut)
            self.parametrized_layers.append(self.BNshortcut)
            self.shortcut_type = '1x1'
        else:
            self.shortcut_type = 'identity'

        for layer in self.parametrized_layers:
            init_layer(layer)

    def forward(self, x):
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu1(out)
        out = self.C2(out)
        out = self.BN2(out)
        short_out = x if self.shortcut_type == 'identity' else self.BNshortcut(self.shortcut(x))
        out = out + short_out
        out = self.relu2(out)
        return out



# Bottleneck block
class BottleneckBlock(nn.Module):
    def __init__(self, indim, outdim, half_res):
        super(BottleneckBlock, self).__init__()
        bottleneckdim = int(outdim/4)
        self.indim = indim
        self.outdim = outdim
        self.C1 = nn.Conv2d(indim, bottleneckdim, kernel_size=1,  bias=False)
        self.BN1 = nn.BatchNorm2d(bottleneckdim)
        self.C2 = nn.Conv2d(bottleneckdim, bottleneckdim, kernel_size=3, stride=2 if half_res else 1,padding=1)
        self.BN2 = nn.BatchNorm2d(bottleneckdim)
        self.C3 = nn.Conv2d(bottleneckdim, outdim, kernel_size=1, bias=False)
        self.BN3 = nn.BatchNorm2d(outdim)

        self.relu = nn.ReLU()
        self.parametrized_layers = [self.C1, self.BN1, self.C2, self.BN2, self.C3, self.BN3]
        self.half_res = half_res


        # if the input number of channels is not equal to the output, then need a 1x1 convolution
        if indim!=outdim:
            self.shortcut = nn.Conv2d(indim, outdim, 1, stride=2 if half_res else 1, bias=False)
            self.parametrized_layers.append(self.shortcut)
            self.shortcut_type = '1x1'
        else:
            self.shortcut_type = 'identity'

        for layer in self.parametrized_layers:
            init_layer(layer)


    def forward(self, x):

        short_out = x if self.shortcut_type == 'identity' else self.shortcut(x)
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu(out)
        out = self.C2(out)
        out = self.BN2(out)
        out = self.relu(out)
        out = self.C3(out)
        out = self.BN3(out)
        out = out + short_out

        out = self.relu(out)
        return out


class ConvNet(nn.Module):
    def __init__(self, depth, flatten = True):
        super(ConvNet,self).__init__()
        trunk = []
        for i in range(depth):
            indim = 3 if i == 0 else 64
            outdim = 64
            B = ConvBlock(indim, outdim, pool = ( i <4 ) ) #only pooling for fist 4 layers
            trunk.append(B)

        if flatten:
            trunk.append(Flatten())

        self.trunk = nn.Sequential(*trunk)
        self.final_feat_dim = 1600

    def forward(self,x):
        out = self.trunk(x)
        return out


class ResNet(nn.Module):
    
    def __init__(self,block,list_of_num_layers, list_of_out_dims, flatten=True,
        no_fc_layer=False, add_bias=False, classifier_type='linear', num_classes=None, lowdim=0):
        
        # list_of_num_layers specifies number of layers in each stage
        # list_of_out_dims specifies number of output channel for each stage
        super(ResNet,self).__init__()
        assert len(list_of_num_layers)==4, 'Can have only four stages'
        conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                            bias=False)
        bn1 = nn.BatchNorm2d(64)

        relu = nn.ReLU()
        pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        init_layer(conv1)
        init_layer(bn1)


        trunk = [conv1, bn1, relu, pool1]

        indim = 64
        for i in range(4):

            for j in range(list_of_num_layers[i]):
                half_res = (i>=1) and (j==0)
                B = block(indim, list_of_out_dims[i], half_res)
                trunk.append(B)
                indim = list_of_out_dims[i]

        if flatten:
            avgpool = nn.AvgPool2d(7)
            trunk.append(avgpool)
            trunk.append(Flatten())
            self.final_feat_dim = indim
        else:
            self.final_feat_dim = [ indim, 7, 7]

        self.lowdim = lowdim
        if self.lowdim > 0:
            trunk.append(torch.nn.Linear(self.final_feat_dim, self.lowdim, bias=False))
            self.final_feat_dim = self.lowdim

        self.trunk = nn.Sequential(*trunk)
        self.classifier_type  = classifier_type
        
        if no_fc_layer is True:
            self.fc = None
        elif classifier_type == 'linear':
            self.fc = nn.Linear(self.final_feat_dim, num_classes)
        elif classifier_type == 'distance-classifier':
            self.fc = distLinear(self.final_feat_dim, num_classes)
        elif classifier_type == 'avg-classifier':
            self.fc = avgLinear(self.final_feat_dim, num_classes)
        elif classifier_type == 'gda':
            self.fc = gaussianDA(self.final_feat_dim, num_classes)
        elif classifier_type == 'ortho-classifier':
            self.fc = orthonormalDistLinear(self.final_feat_dim, num_classes)
        elif classifier_type == 'prin-comp-classifier':
            self.fc = prinCompClassifier(self.final_feat_dim, num_classes)
        else:
            raise ValueError("classifier type not found")

        self.no_fc_layer = no_fc_layer
        self.add_bias = add_bias
        self.scale = nn.Parameter(torch.FloatTensor([1.0]))
        

    def forward(self,x,features_only=False):
        out = self.trunk(x)
        # with torch.no_grad():
        out_norm = torch.norm(out, dim=1, keepdim=True)+0.00001
        out = out.div(out_norm)
        if features_only:
            return out
        if self.add_bias and self.fc is None:
            out = torch.cat([out, 10.*torch.ones((out.size(0), 1), device=out.device)], dim=-1)
        elif self.fc is not None:
            out = self.fc.forward(out)
        return out
    

def Conv4():
    return ConvNet(4)

def Conv6():
    return ConvNet(6)

def ResNet10( flatten = True, **kwargs):
    return ResNet(SimpleBlock, [1,1,1,1],[64,128,256,512], flatten, **kwargs)

def ResNet18( flatten = True, **kwargs):
    return ResNet(SimpleBlock, [2,2,2,2],[64,128,256,512], flatten, **kwargs)

def ResNet34( flatten = True, **kwargs):
    return ResNet(SimpleBlock, [3,4,6,3],[64,128,256,512], flatten, **kwargs)

def ResNet50( flatten = True, **kwargs):
    return ResNet(BottleneckBlock, [3,4,6,3], [256,512,1024,2048], flatten, **kwargs)

def ResNet101( flatten = True, **kwargs):
    return ResNet(BottleneckBlock, [3,4,23,3],[256,512,1024,2048], flatten, **kwargs)