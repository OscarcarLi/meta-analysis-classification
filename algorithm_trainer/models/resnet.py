# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from torch.autograd import Variable
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm


# Batch norm scheme
def get_norm_layer(num_channels, norm_layer='group_norm'):
    if norm_layer=='batch_norm':
        return torch.nn.BatchNorm2d(num_channels,
                    affine=True)
    elif norm_layer=='group_norm':
        num_groups = min(32, num_channels // 2) 
        return torch.nn.GroupNorm(num_channels=num_channels,
                    num_groups=32)
    else:
        raise ValueError('Unrecognized norm type {}'.format(norm_layer))


# Iniitialization
def init_module(model, zero_init_residual):
    
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.GroupNorm):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    # Zero-initialize the last BN in each residual branch,
    # so that the residual branch starts with zeros, and each residual block behaves like an identity.
    # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
    if zero_init_residual:
        for m in model.modules():
            if isinstance(m, Bottleneck):
                nn.init.constant_(m.bn3.weight, 0)
            elif isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)


# Basic ResNet model

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class distLinear(nn.Module):
    def __init__(self, indim, outdim, class_wise_learnable_norm=True):
        super(distLinear, self).__init__()
        self.L = nn.Linear(indim, outdim, bias = False)
        self.class_wise_learnable_norm = class_wise_learnable_norm   
        if self.class_wise_learnable_norm:      
            WeightNorm.apply(self.L, 'weight', dim=0) #split the weight update component to direction and norm      

        if outdim <=200:
            self.scale_factor = 2; 
            #a fixed scale factor to scale the output of cos value into a reasonably large input for softmax
        else:
            self.scale_factor = 10; 
            #in omniglot, a larger scale factor is required to handle >1000 output classes.

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 0.00001)
        if not self.class_wise_learnable_norm:
            L_norm = torch.norm(self.L.weight.data, p=2, dim =1).unsqueeze(1).expand_as(self.L.weight.data)
            self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
        cos_dist = self.L(x_normalized) 
        #matrix product by forward function, but when using WeightNorm, 
        #this also multiply the cosine distance by a class-wise learnable norm
        scores = self.scale_factor * (cos_dist) 

        return scores


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



class BasicBlock(nn.Module):

    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = get_norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = get_norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=None, no_fc_layer=False,
            zero_init_residual=False, classifier_type='linear', add_bias=False):

        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = get_norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.prefc_feature_sz = 512 * block.expansion
        if no_fc_layer is True:
            self.fc = None
        elif classifier_type == 'linear':
            self.fc = nn.Linear(self.prefc_feature_sz, num_classes)
        elif classifier_type == 'distance-classifier':
            self.fc = distLinear(self.prefc_feature_sz, num_classes)
        elif classifier_type == 'gda':
            self.fc = gaussianDA(self.prefc_feature_sz, num_classes)
        else:
            raise ValueError("classifier type not found")

        self.no_fc_layer = no_fc_layer
        self.add_bias = add_bias
        self.scale = nn.Parameter(torch.FloatTensor([1.0]))
        
        init_module(self, zero_init_residual)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                get_norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        if self.add_bias and self.fc is None:
            out = torch.cat([out, 10.*torch.ones((out.size(0), 1), device=out.device)], dim=-1)
        elif self.fc is not None:
            out = self.fc.forward(out)
        return out
    
def ResNet18(**kwargs):
    """
    Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model
