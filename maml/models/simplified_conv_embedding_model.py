from collections import OrderedDict
import torch
import torch.nn.functional as F
import numpy as np

from maml.utils import spectral_norm


def weight_init(module):
    if (isinstance(module, torch.nn.Linear)
        or isinstance(module, torch.nn.Conv2d)):
        torch.nn.init.xavier_normal_(module.weight)
        module.bias.data.zero_()


# the embedding model does not take into account of the labelling task.y?
class RegConvEmbeddingModel(torch.nn.Module):
    def __init__(self, input_size, output_size, modulation_mat_size,
                 hidden_size=128, num_layers=1, num_classes=None, use_label=False,
                 convolutional=False, num_conv=4, num_channels=32, num_channels_max=256,
                 rnn_aggregation=False, linear_before_rnn=False, from_detached_features=False,
                 embedding_pooling='max', batch_norm=True, avgpool_after_conv=True,
                 num_sample_embedding=0, sample_embedding_file='embedding.hdf5', original_conv=False, common_subspace_dim=400,
                 img_size=(1, 28, 28), modulation_mat_spec_norm=5., verbose=False, detached_features_size=None):

        super(RegConvEmbeddingModel, self).__init__()
        self._input_size = input_size
        self._output_size = output_size
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._modulation_mat_size = modulation_mat_size
        self._bidirectional = True
        self._device = 'cpu'
        self._convolutional = convolutional
        self._num_conv = num_conv
        self._num_channels = num_channels
        self._num_channels_max = num_channels_max
        self._batch_norm = batch_norm
        self._img_size = img_size
        self._rnn_aggregation = rnn_aggregation
        self._embedding_pooling = embedding_pooling
        self._linear_before_rnn = linear_before_rnn
        self._embeddings_array = []
        self._num_sample_embedding = num_sample_embedding
        self._sample_embedding_file = sample_embedding_file
        self._avgpool_after_conv = avgpool_after_conv
        self._original_conv = original_conv
        self._reuse = False
        self._modulation_mat_spec_norm = modulation_mat_spec_norm
        self._verbose = verbose
        self._common_subspace_dim = common_subspace_dim
        self._use_label = use_label
        self._from_detached_features = from_detached_features
        self._detached_features_size = detached_features_size 

        if use_label:
            assert num_classes is not None
            self._num_classes = num_classes
            self._label_representations = torch.nn.Embedding(num_embeddings=num_classes,
                embedding_dim=img_size[1] * img_size[2])
        

        conv_list = OrderedDict([])
            
        if self._use_label:
            num_ch = [self._img_size[0] + 1] + [self._num_channels for i in range(self._num_conv)]
        else:
            num_ch = [self._img_size[0]] + [self._num_channels for i in range(self._num_conv)]

        # do not exceed num_channels_max
        num_ch = [min(num_channels_max, ch) for ch in num_ch]
        for i in range(self._num_conv):
            conv_list.update({
                'conv{}'.format(i+1): 
                    torch.nn.Conv2d(num_ch[i], num_ch[i+1], 
                                    (3, 3), stride=2, padding=1)})
            if self._batch_norm:
                # here they uses an extremely small momentum 0.001 as opposed to 0.1
                conv_list.update({
                    'bn{}'.format(i+1):
                        torch.nn.BatchNorm2d(num_ch[i+1], momentum=0.001)})
            conv_list.update({'relu{}'.format(i+1): torch.nn.ReLU(inplace=True)})
        self.conv = torch.nn.Sequential(conv_list)
        self._num_layer_per_conv = len(conv_list) // self._num_conv

        # vectorize the tensor need to know the number of dimensions as rnn input
        # cannot have both linear_before_rnn and avgpool_after_conv
        assert not (self._linear_before_rnn and self._avgpool_after_conv)
        if self._linear_before_rnn:
            # linearly transform the flattened conv features before feeding into rnn
            linear_input_size = self.compute_input_size(
                1, 3, 2, self.conv[self._num_layer_per_conv*(self._num_conv-1)].out_channels)
            rnn_input_size = 128
        elif self._avgpool_after_conv:
            # avg pool across h and w before feeding into rnn
                rnn_input_size = self.conv[self._num_layer_per_conv*(self._num_conv-1)].out_channels
        else:
            # directly use the flattened conv features as input to rnn
            rnn_input_size = self.compute_input_size(
                1, 3, 2, self.conv[self._num_layer_per_conv*(self._num_conv-1)].out_channels)

        if self._rnn_aggregation:
            if self._linear_before_rnn:
                self.linear = torch.nn.Linear(linear_input_size, rnn_input_size)
                self.relu_after_linear = torch.nn.ReLU(inplace=True)
            self.rnn = torch.nn.GRU(rnn_input_size, hidden_size,
                                    num_layers, bidirectional=self._bidirectional)
            embedding_input_size = hidden_size*(2 if self._bidirectional else 1)
        else:
            self.rnn = None
            embedding_input_size = hidden_size
            self.linear = torch.nn.Linear(rnn_input_size, embedding_input_size)
            self.relu_after_linear = torch.nn.ReLU(inplace=True)

        self._modulation_mat_generator = torch.nn.Linear(embedding_input_size, 
                modulation_mat_size[0] * self._common_subspace_dim)

        self._modulation_mat_projection = torch.nn.Linear(self._common_subspace_dim, 
                modulation_mat_size[1])

        self.apply(weight_init)


    def randomize(self, matrix):
        rank = matrix.shape[0]
        noise = torch.randn(rank, rank, device=matrix.device) * (1 / np.sqrt(rank))
        return noise @ matrix

    def forward(self, task, params=None, return_task_embedding=False, detached_features=None, is_training=True):
        if not self._reuse and self._verbose: print('='*8 + ' Emb Model ' + '='*8)
        if params is None:
            params = OrderedDict(self.named_parameters())

        if self._use_label:
            label_embedding = self._label_representations(task.y).view(-1, 1, self._img_size[1], self._img_size[2])
            x = torch.cat((task.x, label_embedding), dim=1)
        else:
            x = task.x

        if not self._reuse and self._verbose: print('input size: {}'.format(x.size()))
        for layer_name, layer in self.conv.named_children():
            # the first 'conv.' comes from the sequential
            weight = params.get('conv.' + layer_name + '.weight', None)
            bias = params.get('conv.' + layer_name + '.bias', None)
            if 'conv' in layer_name:
                x = F.conv2d(x, weight=weight, bias=bias, stride=2, padding=1)
            elif 'relu' in layer_name:
                x = F.relu(x)
            elif 'bn' in layer_name:
                # Question: is the layer.running_mean going to be updated?
                x = F.batch_norm(x, weight=weight, bias=bias,
                                    running_mean=layer.running_mean,
                                    running_var=layer.running_var,
                                    training=True)
            if not self._reuse and self._verbose: print('{}: {}'.format(layer_name, x.size()))
        if self._avgpool_after_conv:
            # average every channel across h * w to reduce to 1 number
            x = x.view(x.size(0), x.size(1), -1)
            if not self._reuse and self._verbose: print('reshape to: {}'.format(x.size()))
            x = torch.mean(x, dim=2)
            if not self._reuse and self._verbose: print('reduce mean: {}'.format(x.size()))

        else:
            # otherwise just vectorize the tensor for each example
            x = task.x.view(task.x.size(0), -1)
            if not self._reuse and self._verbose: print('flatten: {}'.format(x.size()))

            
        if self._rnn_aggregation:
            # LSTM input dimensions are seq_len, batch, input_size
            batch_size = x.size(0)
            h0 = torch.zeros(self._num_layers*(2 if self._bidirectional else 1),
                            batch_size, self._hidden_size, device=self._device)
            if self._linear_before_rnn: 
                x = F.relu(self.linear(x))
            inputs = x.view(1, x.size(0), -1)
            output, hn = self.rnn(inputs, h0)
            # print(output.size(), hn.size())
            output = output.squeeze()
            B, H = output.shape
            output = output.view(B, 2, H // 2)
            # print(output.shape)
            embedding_input = torch.cat(
                [output[-1, 0, :], output[-1, 1, :]], dim=-1).unsqueeze(0)
            # print(embedding_input.size())

        else:
            # every example is now represented as a vector
            # average across examples to produce the representation of the dataset

            embedding_input = F.relu(self.linear(x.mean(0, keepdim=True)))
            
        modulation_mat = self._modulation_mat_generator(embedding_input).reshape(
                            self._modulation_mat_size[0], self._common_subspace_dim)
        
        modulation_mat = self._modulation_mat_projection(modulation_mat)

        # print("modulation_mat:", modulation_mat.size())
        
        if not self._reuse and self._verbose: print('modulation mat {}'.format(
                modulation_mat.size()))

        modulation_mat = spectral_norm(modulation_mat, device=self._device,
            limit = self._modulation_mat_spec_norm)
        
        if not self._reuse and self._verbose: print('='*27)
        self._reuse = True

        # if is_training:
        modulation_mat = self.randomize(modulation_mat)
        modulation_mat = torch.svd(modulation_mat)[-1].t()

        if return_task_embedding:
            return (modulation_mat, None), embedding_input
        else:
            return (modulation_mat, None)


    def to(self, device, **kwargs):
        self._device = device
        super(RegConvEmbeddingModel, self).to(device, **kwargs)