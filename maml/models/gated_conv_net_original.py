from collections import OrderedDict

import torch
import torch.nn.functional as F
import numpy as np

from maml.models.model import Model
from torch.nn.utils import spectral_norm

def weight_init(module):

    if (isinstance(module, torch.nn.Linear)
        or isinstance(module, torch.nn.Conv2d)):
        torch.nn.init.kaiming_normal_(module.weight)
        if module.bias:
            module.bias.data.zero_()

    # elif isinstance(module, torch.nn.BatchNorm2d):
    #     module.weight.data.fill_(1)
    #     module.bias.data.zero_()
    


class RegConvModel(Model):
    """
    NOTE: difference to tf implementation: batch norm scaling is enabled here
    TODO: enable 'non-transductive' setting as per
          https://arxiv.org/abs/1803.02999
    """
    def __init__(self, input_channels, output_size, modulation_mat_rank, num_channels=64,
                 kernel_size=3, padding=1, nonlinearity=F.relu,
                 use_max_pool=False, img_side_len=28, verbose=False):
        super(RegConvModel, self).__init__()
        self._input_channels = input_channels
        self._output_size = output_size
        self._modulation_mat_rank = modulation_mat_rank
        self._num_channels = num_channels
        self._kernel_size = kernel_size
        self._nonlinearity = nonlinearity
        self._use_max_pool = use_max_pool
        self._padding = padding
        # When affine=False the output of BatchNorm is equivalent to considering gamma=1 and beta=0 as constants.
        self._bn_affine = False
        # reuse is for checking the model architecture
        self._reuse = False
        self._verbose = verbose

        # use 2 by 2 max_pool then use conv_stride = 1
        self._conv_stride = 1
        self._padding = 1
        self._kernel_size = 3
        # self._features_size = 1 # _features_size = 1 is clearly a bug in the original code
        self.features = torch.nn.Sequential(OrderedDict([
            ('layer1_conv', torch.nn.Conv2d(self._input_channels,
                                            self._num_channels,
                                            self._kernel_size,
                                            stride=self._conv_stride,
                                            padding=self._padding)),
            ('layer1_bn', torch.nn.BatchNorm2d(self._num_channels,
                                                affine=self._bn_affine,
                                                momentum=0.001)),
            ('layer1_relu', torch.nn.ReLU(inplace=True)),
            ('layer1_max_pool', torch.nn.MaxPool2d(kernel_size=2,
                                                    stride=2)),
            ('layer2_conv', torch.nn.Conv2d(self._num_channels,
                                            self._num_channels,
                                            self._kernel_size,
                                            stride=self._conv_stride,
                                            padding=self._padding)),
            ('layer2_bn', torch.nn.BatchNorm2d(self._num_channels,
                                                affine=self._bn_affine,
                                                momentum=0.001)),
            ('layer2_relu', torch.nn.ReLU(inplace=True)),
            ('layer2_max_pool', torch.nn.MaxPool2d(kernel_size=2,
                                                    stride=2)),
            ('layer3_conv', torch.nn.Conv2d(self._num_channels,
                                            self._num_channels,
                                            self._kernel_size,
                                            stride=self._conv_stride,
                                            padding=self._padding)), 
            ('layer3_bn', torch.nn.BatchNorm2d(self._num_channels,
                                                affine=self._bn_affine,
                                                momentum=0.001)),
            ('layer3_relu', torch.nn.ReLU(inplace=True)),
            ('layer3_max_pool', torch.nn.MaxPool2d(kernel_size=2,
                                                    stride=2)),
            ('layer4_conv', torch.nn.Conv2d(self._num_channels,
                                            self._num_channels,
                                            self._kernel_size,
                                            stride=self._conv_stride,
                                            padding=self._padding)),
            ('layer4_bn', torch.nn.BatchNorm2d(self._num_channels,
                                                affine=self._bn_affine,
                                                momentum=0.001)),
            ('layer4_max_pool', torch.nn.MaxPool2d(kernel_size=2,
                                                    stride=2)), 
        ]))
        
        self.classifier = torch.nn.Sequential(OrderedDict([
            ('fully_connected', torch.nn.Linear(self._modulation_mat_rank,
                                                self._output_size))
        ]))
        self.apply(weight_init)

    def forward(self, batch, modulation, update_params=None, training=True):
        if not self._reuse and self._verbose: print('='*10 + ' Model ' + '='*10)
        params = OrderedDict(self.named_parameters())

        x = batch
        
        modulation_mat, modulation_bias = modulation

        if not self._reuse and self._verbose: print('input size: {}'.format(x.size()))
        for layer_name, layer in self.features.named_children():
            weight = params.get('features.' + layer_name + '.weight', None)
            bias = params.get('features.' + layer_name + '.bias', None)
            if 'conv' in layer_name:
                x = F.conv2d(x, weight=weight, bias=bias,
                             stride=self._conv_stride, padding=self._padding)
            elif 'bn' in layer_name:
                x = F.batch_norm(x, weight=weight, bias=bias,
                                 running_mean=layer.running_mean,
                                 running_var=layer.running_var,
                                 training=training)
            elif 'max_pool' in layer_name:
                x = F.max_pool2d(x, kernel_size=2, stride=2)
            elif 'relu' in layer_name:
                x = F.relu(x)
            elif 'fully_connected' in layer_name:
                # we have reached the last layer
                break
            else:
                raise ValueError('Unrecognized layer {}'.format(layer_name))
            if not self._reuse and self._verbose: print('{}: {}'.format(layer_name, x.size()))

        # below the conv maps are flattened
        x = x.view(x.size(0), -1)
        x = F.linear(x, weight=modulation_mat,
                        bias=modulation_bias)

        
        if update_params is None:
            logits = F.linear(x, weight=params['classifier.fully_connected.weight'],
            bias=params['classifier.fully_connected.bias'])
                
        else:
            # we have to use update_params here instead of params
            logits = F.linear(x, weight=update_params['classifier.fully_connected.weight'],
                         bias=update_params['classifier.fully_connected.bias'])

        if not self._reuse and self._verbose: print('logits size: {}'.format(logits.size()))
        if not self._reuse and self._verbose: print('='*27)
        self._reuse = True
        return logits







class ImpRegConvModel(Model):
    """
    NOTE: difference to tf implementation: batch norm scaling is enabled here
    TODO: enable 'non-transductive' setting as per
          https://arxiv.org/abs/1803.02999
    """
    def __init__(self, input_channels, output_size, modulation_mat_rank, num_channels=64,
                 kernel_size=3, padding=1, nonlinearity=F.relu,
                 use_max_pool=False, img_side_len=28, verbose=False, normalize_norm=0.):
        super(ImpRegConvModel, self).__init__()
        self._input_channels = input_channels
        self._output_size = output_size
        self._modulation_mat_rank = modulation_mat_rank
        self._num_channels = num_channels
        self._kernel_size = kernel_size
        self._nonlinearity = nonlinearity
        self._use_max_pool = use_max_pool
        self._normalize_norm = normalize_norm
        self._padding = padding
        # When affine=False the output of BatchNorm is equivalent to considering gamma=1 and beta=0 as constants.
        self._bn_affine = False
        # reuse is for checking the model architecture
        self._reuse = False
        self._verbose = verbose

        # use 2 by 2 max_pool then use conv_stride = 1
        self._conv_stride = 1
        self._padding = 1
        self._kernel_size = 3
        # self._features_size = 1 # _features_size = 1 is clearly a bug in the original code
        self.features = torch.nn.Sequential(OrderedDict([
            ('layer1_conv', torch.nn.Conv2d(self._input_channels,
                                            self._num_channels,
                                            self._kernel_size,
                                            stride=self._conv_stride,
                                            padding=self._padding,
                                            bias=False)),
            ('layer1_bn', torch.nn.BatchNorm2d(self._num_channels,
                                                affine=self._bn_affine,
                                                momentum=0.001)),
            ('layer1_relu', torch.nn.ReLU(inplace=True)),
            ('layer1_max_pool', torch.nn.MaxPool2d(kernel_size=2,
                                                    stride=2)),
            ('layer2_conv', torch.nn.Conv2d(self._num_channels,
                                            self._num_channels,
                                            self._kernel_size,
                                            stride=self._conv_stride,
                                            padding=self._padding,
                                            bias=False)),
            ('layer2_bn', torch.nn.BatchNorm2d(self._num_channels,
                                                affine=self._bn_affine,
                                                momentum=0.001)),
            ('layer2_relu', torch.nn.ReLU(inplace=True)),
            ('layer2_max_pool', torch.nn.MaxPool2d(kernel_size=2,
                                                    stride=2)),
            ('layer3_conv', torch.nn.Conv2d(self._num_channels,
                                            self._num_channels,
                                            self._kernel_size,
                                            stride=self._conv_stride,
                                            padding=self._padding,
                                            bias=False)),
            ('layer3_bn', torch.nn.BatchNorm2d(self._num_channels,
                                                affine=self._bn_affine,
                                                momentum=0.001)),
            ('layer3_relu', torch.nn.ReLU(inplace=True)),
            ('layer3_max_pool', torch.nn.MaxPool2d(kernel_size=2,
                                                    stride=2)),
            ('layer4_conv', torch.nn.Conv2d(self._num_channels,
                                            self._num_channels,
                                            self._kernel_size,
                                            stride=self._conv_stride,
                                            padding=self._padding,
                                            bias=False)),
            ('layer4_bn', torch.nn.BatchNorm2d(self._num_channels,
                                                affine=self._bn_affine,
                                                momentum=0.001)),
            ('layer4_max_pool', torch.nn.MaxPool2d(kernel_size=2,
                                                    stride=2)),
        ]))
        
        self.apply(weight_init)

        

        
    def compute_kernel_matrix(self, X, Y, sigma_list=[]):
        
        assert(X.size(1) == Y.size(1))
        Z = torch.cat((X, Y), 0)
        ZZT = torch.mm(Z, Z.t())
        diag_ZZT = torch.diag(ZZT).unsqueeze(1)
        Z_norm_sqr = diag_ZZT.expand_as(ZZT)
        exponent = Z_norm_sqr - 2 * ZZT + Z_norm_sqr.t()

        kernel_matrices = [X @ Y.T]
        for sigma in sigma_list:
            gamma = 1.0 / (2 * sigma**2)
            kernel_matrices.append(torch.exp(-gamma * exponent)[:X.size(0), X.size(0):])

        return kernel_matrices
        

    def forward(self, batch, modulation, training=True, features=None, only_features=False):
        
        if not self._reuse and self._verbose: print('='*10 + ' Model ' + '='*10)
        
        x = batch
        if not self._reuse and self._verbose: print('input size: {}'.format(x.size()))
    
        x = self.features(x) 
        x = x.view(x.size(0), -1)
        if not self._reuse and self._verbose: print('features size: {}'.format(x.size()))

        # if not self._reuse:
        #     print("Before Modulation")
        #     print(x.shape, torch.norm(x, p=2, dim=1))
    
        # x_post_mod = []
        # d = modulation[0].size(1) // len(kernel_matrices)
        # for i in range(len(modulation)):
        #     x_post_mod.append(kernel_matrices[i] @ modulation[0][:, i*d:(i+1)*d])
        # x = torch.cat(x_post_mod, dim=-1)


        x_post_mod = []
        for mod in modulation:
            x_post_mod.append(x @ mod.T)
        x = x_post_mod
        
        # if not self._reuse:
        #     print("After modulation")
        #     print([(z.shape, torch.norm(z, p=2, dim=1)) for z in x])

        x = [torch.cat([z, 10.*torch.ones((z.size(0), 1), device=z.device)], dim=-1)
                for z in x]

        return x