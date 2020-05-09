from collections import OrderedDict

import torch
import torch.nn.functional as F

from maml.models.model import Model
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


def weight_init(module):
    if (isinstance(module, torch.nn.Linear)
        or isinstance(module, torch.nn.Conv2d)):
        torch.nn.init.kaiming_normal_(module.weight)
        module.bias.data.zero_()


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
            ('layer1_relu', torch.nn.ReLU(inplace=True)),
            ('layer1_bn', torch.nn.BatchNorm2d(self._num_channels,
                                                affine=self._bn_affine,
                                                momentum=0.001)),
            ('layer1_max_pool', torch.nn.MaxPool2d(kernel_size=2,
                                                    stride=2)),
            ('layer2_conv', torch.nn.Conv2d(self._num_channels,
                                            self._num_channels,
                                            self._kernel_size,
                                            stride=self._conv_stride,
                                            padding=self._padding)),
            ('layer2_relu', torch.nn.ReLU(inplace=True)),
            ('layer2_bn', torch.nn.BatchNorm2d(self._num_channels,
                                                affine=self._bn_affine,
                                                momentum=0.001)),
            ('layer2_max_pool', torch.nn.MaxPool2d(kernel_size=2,
                                                    stride=2)),
            ('layer3_conv', torch.nn.Conv2d(self._num_channels,
                                            self._num_channels,
                                            self._kernel_size,
                                            stride=self._conv_stride,
                                            padding=self._padding)),
            ('layer3_relu', torch.nn.ReLU(inplace=True)),
            
            ('layer3_bn', torch.nn.BatchNorm2d(self._num_channels,
                                                affine=self._bn_affine,
                                                momentum=0.001)),
            
            ('layer3_max_pool', torch.nn.MaxPool2d(kernel_size=2,
                                                    stride=2)),
            ('layer4_conv', torch.nn.Conv2d(self._num_channels,
                                            self._num_channels,
                                            self._kernel_size,
                                            stride=self._conv_stride,
                                            padding=self._padding)),
            ('layer4_relu', torch.nn.ReLU(inplace=True)),
            ('layer4_bn', torch.nn.BatchNorm2d(self._num_channels,
                                                affine=self._bn_affine,
                                                momentum=0.001)),
            ('layer4_max_pool', torch.nn.MaxPool2d(kernel_size=2,
                                                    stride=1)), 
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
                                            padding=self._padding)),
            ('layer1_relu', torch.nn.ReLU(inplace=True)),
            ('layer1_bn', torch.nn.BatchNorm2d(self._num_channels,
                                                affine=self._bn_affine,
                                                momentum=0.001)),
            ('layer1_max_pool', torch.nn.MaxPool2d(kernel_size=2,
                                                    stride=2)),
            ('layer2_conv', torch.nn.Conv2d(self._num_channels,
                                            self._num_channels,
                                            self._kernel_size,
                                            stride=self._conv_stride,
                                            padding=self._padding)),
            ('layer2_relu', torch.nn.ReLU(inplace=True)),
            ('layer2_bn', torch.nn.BatchNorm2d(self._num_channels,
                                                affine=self._bn_affine,
                                                momentum=0.001)),
            ('layer2_max_pool', torch.nn.MaxPool2d(kernel_size=2,
                                                    stride=2)),
            ('layer3_conv', torch.nn.Conv2d(self._num_channels,
                                            self._num_channels,
                                            self._kernel_size,
                                            stride=self._conv_stride,
                                            padding=self._padding)),
            ('layer3_relu', torch.nn.ReLU(inplace=True)),
            
            ('layer3_bn', torch.nn.BatchNorm2d(self._num_channels,
                                                affine=self._bn_affine,
                                                momentum=0.001)),
            
            ('layer3_max_pool', torch.nn.MaxPool2d(kernel_size=2,
                                                    stride=2)),
            ('layer4_conv', torch.nn.Conv2d(self._num_channels,
                                            self._num_channels,
                                            self._kernel_size,
                                            stride=self._conv_stride,
                                            padding=self._padding)),
            ('layer4_relu', torch.nn.ReLU(inplace=True)),
            ('layer4_bn', torch.nn.BatchNorm2d(self._num_channels,
                                                affine=self._bn_affine,
                                                momentum=0.001)),
            ('layer4_max_pool', torch.nn.MaxPool2d(kernel_size=2,
                                                    stride=1)), 
        ]))
        
        self.apply(weight_init)

        # self.classifier = torch.nn.Sequential(OrderedDict([
        #     ('fully_connected', torch.nn.Linear(self._modulation_mat_rank + 1,
        #                                         self._output_size))
        # ]))
        
    def forward(self, batch, modulation, training=True, only_features=False):
        if not self._reuse and self._verbose: print('='*10 + ' Model ' + '='*10)
        params = OrderedDict(self.named_parameters())

        x = batch
        
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
            if not self._reuse and self._verbose: print('{}: {}, norm : {}'.format(layer_name, x.size(), torch.norm(x.view(x.size(0), -1), p=2, dim=1).mean(0)))


        if only_features:
            return x
            
        # below the conv maps are flattened
        x = x.view(x.size(0), -1)

        

        if not self._reuse:
            print("Before Modulation")
            print(torch.norm(x, p=2, dim=1))


        x = F.linear(x, weight=modulation, bias=None) # dont use modulation bias

        # n_samples = x.size(0) // 5
        # if n_samples > 1:
        #     m = 0
        #     print(n_samples)
        #     for i in range(5):
        #         m+= cosine_similarity(x.detach().cpu().numpy()[i*n_samples:(i+1)*n_samples, :]).mean()
        #     print("avg intra class similarity:", m/5)

        #     # y = x[np.random.permutation(x.size(0))]
        #     y = x
        #     m = []
        #     for i in range(5):
        #         m.append((y.detach().cpu().numpy()[i*n_samples:(i+1)*n_samples, :]).mean(0))
        #     m = np.array(m)
        #     print("avg inter class similarity:", cosine_similarity(m).mean())


        if not self._reuse:
            print("After modulation")
            print(torch.norm(x, p=2, dim=1))

        if self._normalize_norm > 0.:
            max_norm = torch.max(
                torch.norm(x, p=2, dim=1))
            x = x.div(max_norm) * self._normalize_norm

        if not self._reuse and self._normalize_norm > 0.:
            print("After Normalize")
            print(torch.norm(x, p=2, dim=1))

        x = torch.cat([x, 10.*torch.ones((x.size(0), 1), device=x.device)], dim=-1)
        # pad with 10 to allow higher bias in inner solver
        
        self._reuse = True

        return x


        # pad with 10 to allow higher bias in inner solver

        # if update_params is None:
        #     logits = F.linear(x, weight=params['classifier.fully_connected.weight'],
        #     bias=params['classifier.fully_connected.bias'])
                
        # else:
        #     # we have to use update_params here instead of params
        #     logits = F.linear(x, weight=update_params['classifier.fully_connected.weight'],
        #                  bias=update_params['classifier.fully_connected.bias'])

        # if not self._reuse and self._verbose: print('logits size: {}'.format(logits.size()))
        # if not self._reuse and self._verbose: print('='*27)