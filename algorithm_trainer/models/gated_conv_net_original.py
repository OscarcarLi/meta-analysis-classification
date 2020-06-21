from collections import OrderedDict
import math
import torch
import torch.nn.functional as F
import torch.nn as nn

from algorithm_trainer.models.model import Model


def weight_init(module):
    if isinstance(module, nn.Conv2d):
        n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
        module.weight.data.normal_(0, math.sqrt(2. / n))  
    if (isinstance(module, torch.nn.Linear)
        or isinstance(module, torch.nn.Conv2d)):
        torch.nn.init.kaiming_normal_(module.weight)
        module.bias.data.zero_()
    elif isinstance(module, nn.GroupNorm):
        module.weight.data.fill_(1)
        module.bias.data.zero_()


def get_norm_layer(num_channels, norm_layer='batch_norm'):
    if norm_layer=='batch_norm':
        return torch.nn.BatchNorm2d(num_channels,
                    affine=False, momentum=0.001)
    elif norm_layer=='group_norm':
        return torch.nn.GroupNorm(num_channels=num_channels,
                    num_groups=32)
    else:
        raise ValueError('Unrecognized norm type {}'.format(norm_layer))


class Conv64(Model):
    """
    NOTE: difference to tf implementation: batch norm scaling is enabled here
    TODO: enable 'non-transductive' setting as per
          https://arxiv.org/abs/1803.02999
    """
    def __init__(self, num_channels=64, verbose=False, num_classes=None,
        retain_activation=False, use_group_norm=False, add_bias=False,
        no_fc_layer=False):
        
        super(Conv64, self).__init__()

        self._num_channels = num_channels
        self._retain_activation = retain_activation
        self._use_group_norm = use_group_norm
        self._add_bias = add_bias 
        self._no_fc_layer = no_fc_layer
        self._num_classes = num_classes
        self._verbose = verbose
        

        # fixed constants (for now)
        self._kernel_size = 3
        self._nonlinearity = F.relu
        self._padding = 1
        self._input_channels = 3
        self._conv_stride = 1
        self._reuse = False
        
        print("add_bias to output features : ", self._add_bias)

        self.features = torch.nn.Sequential(OrderedDict([
            ('layer1_conv', torch.nn.Conv2d(self._input_channels,
                                            self._num_channels,
                                            self._kernel_size,
                                            stride=self._conv_stride,
                                            padding=self._padding)),
            ('layer1_norm', get_norm_layer(self._num_channels, 
                    'group_norm' if self._use_group_norm else 'batch_norm')),
            ('layer1_relu', torch.nn.ReLU(inplace=True)),
            ('layer1_max_pool', torch.nn.MaxPool2d(kernel_size=2,
                                                    stride=2)),
            ('layer2_conv', torch.nn.Conv2d(self._num_channels,
                                            self._num_channels,
                                            self._kernel_size,
                                            stride=self._conv_stride,
                                            padding=self._padding)),
            ('layer2_norm', get_norm_layer(self._num_channels, 
                    'group_norm' if self._use_group_norm else 'batch_norm')),
            ('layer2_relu', torch.nn.ReLU(inplace=True)),
            ('layer2_max_pool', torch.nn.MaxPool2d(kernel_size=2,
                                                    stride=2)),
            ('layer3_conv', torch.nn.Conv2d(self._num_channels,
                                            self._num_channels,
                                            self._kernel_size,
                                            stride=self._conv_stride,
                                            padding=self._padding)),
            ('layer3_norm', get_norm_layer(self._num_channels, 
                    'group_norm' if self._use_group_norm else 'batch_norm')),
            ('layer3_relu', torch.nn.ReLU(inplace=True)),
            ('layer3_max_pool', torch.nn.MaxPool2d(kernel_size=2,
                                                    stride=2)),
            ('layer4_conv', torch.nn.Conv2d(self._num_channels,
                                            self._num_channels,
                                            self._kernel_size,
                                            stride=self._conv_stride,
                                            padding=self._padding)),
            ('layer4_norm', get_norm_layer(self._num_channels, 
                    'group_norm' if self._use_group_norm else 'batch_norm')),
        ]))

        self.scale = nn.Parameter(torch.FloatTensor([1.0]))
        self.add_module(self.scale)

        if self._retain_activation:
            self.features.add_module('layer4_relu', torch.nn.ReLU(inplace=True))
            
        self.features.add_module('layer4_max_pool', 
                torch.nn.MaxPool2d(kernel_size=2, stride=2))

        if self._no_fc_layer is False:
            self.fc = torch.nn.Linear(
                self.num_channels * 25, self._num_classes)
        
        self.apply(weight_init)


    def forward(self, x):
        
        if not self._reuse and self._verbose: print('input size: {}'.format(x.size()))
        x = self.features(x)
        
    
        # conv maps are flattened
        x = x.view(x.size(0), -1)
        if not self._reuse and self._verbose: print('after flatten: {}'.format(x.size()))
        
        if not self._reuse and self._verbose: 
            print('size: {}, norm : {}'.format(x.size(), 
                torch.norm(x.view(x.size(0), -1), p=2, dim=1).mean(0)))

        if self._add_bias and self._no_fc_layer:
            x = torch.cat([x, 10.*torch.ones((x.size(0), 1), device=x.device)], dim=-1)
        elif self._self._no_fc_layer is False:
            x = self.fc(x)
            if not self._reuse and self._verbose: 
                print('logits: {}'.format(x.size()))
        
            
        self._reuse = True
        return x