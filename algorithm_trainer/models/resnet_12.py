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

# This ResNet network was designed following the practice of the following papers:
# TADAM: Task dependent adaptive metric for improved few-shot learning (Oreshkin et al., in NIPS 2018) and
# A Simple Neural Attentive Meta-Learner (Mishra et al., in ICLR 2018).




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


    # def update_L(self, xg, yg, x, y):

    #         c_mat = []
    #         c_mat_g = []
    #         for c in np.arange(self.outdim):
    #             c_feat = torch.mean(x[y==c, :], dim=0)
    #             c_feat_g = torch.mean(xg[yg==c, :], dim=0) 
    #             c_mat.append(c_feat)        
    #             c_mat_g.append(c_feat_g)        
    #         c_mat = torch.stack(c_mat, dim=0)
    #         c_mat_g = torch.stack(c_mat_g, dim=0)
    #         self.L = self.lambd * c_mat_g + (1. - self.lambd) * c_mat
            

    # def update_Lg(self, x, y):

    #     # print("recvd:", "x:", x.shape, "y:", y.shape)
    #     with torch.no_grad():
    #         # self.Lg.weight = self.L.detach()

    #         for c in np.unique(y.cpu().numpy()):
    #             c_feat = torch.mean(x[y==c, :], dim=0)
    #             # c_feat = c_feat.div(torch.norm(c_feat, p=2)+ 0.00001)
    #             self.Lg.weight[c, :] = self.gamma * self.Lg.weight[c, :] + (1. - self.gamma) * c_feat


    # def update_Lg_full(self, x, y):

    #     # print("recvd:", "x:", x.shape, "y:", y.shape)
    #     with torch.no_grad():
    #         for c in np.unique(y.cpu().numpy()):
    #             c_feat = torch.sum(x[y==c, :], dim=0)
    #             self.Lg.weight[c, :] = self.Lg.weight[c, :] + c_feat
    #             self.class_count[c] += x[y==c, :].shape[0]


    # def divide_Lg(self):

    #     for c in self.class_count:
    #         self.Lg.weight[c, :] = self.Lg.weight[c, :].div(self.class_count[c])
    #         # print("c:", c, self.class_count[c])
    #         self.class_count[c] = 0
    #     # self.Lglinear.weight = torch.nn.Parameter(self.Lg.weight)
        
        


    # def project_Lg(self):

    #     Lg_norm = torch.norm(self.Lg.weight.data, p=2, dim =1).unsqueeze(1).expand_as(self.Lg.weight.data)
    #     self.Lg.weight.data = self.Lg.weight.data.div(Lg_norm + 0.00001)
        

    # def compute_loss(self):   

    #     # I = torch.eye(self.outdim, device=self.L.device)
    #     # print("L:", self.L)
    #     # print("Lg:", self.Lg.weight.t())
        
    #     # print(self.L @ self.Lg.weight.t())
    #     # print(torch.sum((self.L @ self.Lg.weight.t())**2))

    #     # L_n = torch.norm(self.L, dim=1, p=2)
    #     # Lg_n = torch.norm(self.Lg.weight, dim=1, p=2)

    #     # loss = torch.sum((L_n * Lg_n - torch.diag(self.L @ self.Lg.weight.t()))**2) 
    #     loss = torch.sum(torch.diag(1.0 - self.L @ self.Lg.weight.T))
    #     loss /= self.L.shape[0]
    #     # loss /= (self.L.shape[0] * np.sqrt(self.L.shape[1])) 
    #     return loss






def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.0, drop_block=False, block_size=1):
        
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        # self.num_batches_tracked = 0
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        self.drop_block = drop_block
        self.block_size = block_size
        self.DropBlock = DropBlock(block_size=self.block_size)



    def forward(self, x):
        self.num_batches_tracked += 1

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.maxpool(out)
        
        if self.drop_rate > 0:
            if self.drop_block == True:
                feat_size = out.size()[2]
                keep_rate = max(1.0 - self.drop_rate / (20*2000) * (self.num_batches_tracked), 1.0 - self.drop_rate)
                gamma = (1 - keep_rate) / self.block_size**2 * feat_size**2 / (feat_size - self.block_size + 1)**2
                out = self.DropBlock(out, gamma=gamma)
            else:
                out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)

        return out


class ResNet(nn.Module):

    def __init__(self, block, keep_prob=1.0, avg_pool=False, drop_rate=0.0, dropblock_size=5,
            num_classes=200, classifier_type='distance-classifier', add_bias=False,
            projection=True, classifier_metric='cosine', lambd=0.):

        self.inplanes = 3
        super(ResNet, self).__init__()
        self.projection = projection
        print("Unit norm projection is ", self.projection)
        self.layer1 = self._make_layer(block, 64, stride=2, drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, 160, stride=2, drop_rate=drop_rate)
        self.layer3 = self._make_layer(block, 320, stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        self.layer4 = self._make_layer(block, 640, stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        if avg_pool:
            self.avgpool = nn.AvgPool2d(dropblock_size, stride=1)
        self.keep_prob = keep_prob
        self.keep_avg_pool = avg_pool
        self.dropout = nn.Dropout(p=1 - self.keep_prob, inplace=False)
        self.drop_rate = drop_rate
        print("Average pooling: ", self.keep_avg_pool)


        # classifier creation
        self.final_feat_dim = 640
        self.classifier_type  = classifier_type
        self.num_classes = num_classes
        self.no_fc_layer = (classifier_type == "no-classifier")
        
        if self.no_fc_layer is True:
            self.fc = None
        elif classifier_type == 'linear':
            self.fc = nn.Linear(self.final_feat_dim, num_classes)
            self.fc.bias.data.fill_(0)
        elif classifier_type == 'distance-classifier':
            self.fc = distLinear(self.final_feat_dim, num_classes)
        elif classifier_type == 'cvx-classifier':
            self.fc = CvxClasifier(self.final_feat_dim, num_classes, 
            metric=classifier_metric, lambd_start=lambd, lambd_end=lambd, projection=self.projection)
        else:
            raise ValueError("classifier type not found")

        self.add_bias = add_bias
        self.scale_factor = nn.Parameter(torch.FloatTensor([10.0]))
        

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)



    def _make_layer(self, block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, drop_rate, drop_block, block_size))
        self.inplanes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, features_only=True):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.keep_avg_pool:
            x = self.avgpool(x)
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


def resnet12(keep_prob=1.0, avg_pool=True, **kwargs):
    """Constructs a ResNet-12 model.
    """
    model = ResNet(BasicBlock, keep_prob=keep_prob, avg_pool=avg_pool, **kwargs)
    return model



