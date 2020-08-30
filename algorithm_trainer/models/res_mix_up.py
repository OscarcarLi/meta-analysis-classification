import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch.nn.utils.weight_norm import WeightNorm

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def mixup_data(x, y, lam):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
   
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    if torch.cuda.is_available():
        index = index.cuda()
    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam

class distLinear(nn.Module):
    def __init__(self, indim, outdim):
        super(distLinear, self).__init__()
        self.L = nn.Linear( indim, outdim, bias = False)
        self.class_wise_learnable_norm = True  #See the issue#4&8 in the github 
        if self.class_wise_learnable_norm:      
            WeightNorm.apply(self.L, 'weight', dim=0) #split the weight update component to direction and norm      

        if outdim <=200:
            self.scale_factor = 2; #a fixed scale factor to scale the output of cos value into a reasonably large input for softmax
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

    def __init__(self, indim, outdim, momentum=0.95):
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

    def K(self, a, b):
        """ linear kernel
        """
        return self.M(a) @ self.M(b).T 

    def forward(self, x):
        Lg_norm = torch.norm(self.Lg.weight.data, p=2, dim =1).unsqueeze(1).expand_as(self.Lg.weight.data)
        self.Lg.weight.data = self.Lg.weight.data.div(Lg_norm + 0.00001)
        # scores = self.scale_factor * (self.gbeta * (self.K(x, self.L)) + self.K(x, self.Lg.weight))
        scores = self.scale_factor * ((self.K(x, self.L) - self.K(x, self.L.detach()))  + self.K(x, self.Lg.weight))
        return scores

    def update_L(self, x, y):

        for c in np.unique(y.cpu().numpy()):
            c_feat = torch.mean(x[y==c, :], dim=0)
            c_feat = c_feat.div(torch.norm(c_feat, p=2)+ 0.00001)
            self.L.data[c, :] = c_feat


    def update_Lg(self, x, y):

        with torch.no_grad():
            for c in np.unique(y.cpu().numpy()):
                c_feat = torch.mean(x[y==c, :], dim=0)
                c_feat = c_feat.div(torch.norm(c_feat, p=2)+ 0.00001)
                self.Lg.weight[c, :] = self.momentum * self.Lg.weight[c, :] + (1 - self.momentum) * c_feat


    def update_Lg_full(self, x, y):

        with torch.no_grad():
            for c in np.unique(y.cpu().numpy()):
                c_feat = torch.sum(x[y==c, :], dim=0) / 600.
                self.Lg.weight[c, :] = self.Lg.weight[c, :] + c_feat
        

        





class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
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

    def __init__(self, block, layers, num_classes=200, zero_init_residual=False,
            classifier_type='distance-classifier', no_fc_layer=False, add_bias=False):
        
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        
        # classifier creation
        self.final_feat_dim = 512 * block.expansion
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
        


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, target=None, mixup=False, mixup_hidden = True, mixup_alpha=None, lam=0.4, features_only=False):
        if target is not None: 
            if mixup_hidden:
                layer_mix = random.randint(0,5)
            elif mixup:
                layer_mix = 0
            else:
                layer_mix = None
            
            out = x
            
            if layer_mix == 0:
                out, target_a, target_b, lam = mixup_data(out, target, lam=lam)
                  
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
    
            if layer_mix == 1:
                out, target_a, target_b, lam = mixup_data(out, target, lam=lam)

            out = self.layer2(out)
    
            if layer_mix == 2:
                out, target_a, target_b, lam = mixup_data(out, target, lam=lam)

            out = self.layer3(out)
            
            if layer_mix == 3:
                out, target_a, target_b, lam = mixup_data(out, target, lam=lam)

            out = self.layer4(out)
            
            if layer_mix == 4:
                out, target_a, target_b, lam = mixup_data(out, target, lam=lam)

            out = self.avgpool(out)
            out = out.view(out.size(0), -1)
            out1 = self.fc.forward(out)
            
            if layer_mix == 5:
                out, target_a, target_b, lam = mixup_data(out, target, lam=lam)
            
            # hypersphere projection
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
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = self.avgpool(out)
            out = out.view(out.size(0), -1)   

            # hypersphere projection
            out_norm = torch.norm(out, dim=1, keepdim=True)+0.00001
            out = out.div(out_norm)

            if features_only:
                return out
            if self.add_bias and self.fc is None:
                out = torch.cat([out, 10.*torch.ones((out.size(0), 1), device=out.device)], dim=-1)
            elif self.fc is not None:
                out = self.fc.forward(out)

            return out


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model
