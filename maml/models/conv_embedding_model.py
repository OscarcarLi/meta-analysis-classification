from collections import OrderedDict
import torch
import torch.nn.functional as F
import numpy as np


# the embedding model does not take into account of the labelling task.y?
class ConvEmbeddingModel(torch.nn.Module):
    def __init__(self, img_size, modulation_dims,
                 use_label=False, num_classes=None,
                 hidden_size=128, num_layers=1,
                 convolutional=False, num_conv=4, num_channels=32, num_channels_max=256,
                 rnn_aggregation=False, linear_before_rnn=False, 
                 embedding_pooling='max', batch_norm=True, avgpool_after_conv=True,
                 num_sample_embedding=0, sample_embedding_file='embedding.hdf5',
                 verbose=False):

        super(ConvEmbeddingModel, self).__init__()

        self._use_label = use_label
        
        if use_label:
            assert num_classes is not None
            self._num_classes = num_classes
            self._label_representations = torch.nn.Embedding(num_embeddings=num_classes,
             embedding_dim=img_size[1] * img_size[2])
        
        self._img_size = img_size
        self._input_size = img_size[0] * img_size[1] * img_size[2]
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._modulation_dims = modulation_dims
        self._bidirectional = True
        self._device = 'cpu'
        self._convolutional = convolutional
        self._num_conv = num_conv
        self._num_channels = num_channels
        self._num_channels_max = num_channels_max
        self._batch_norm = batch_norm
        self._rnn_aggregation = rnn_aggregation
        self._embedding_pooling = embedding_pooling
        self._linear_before_rnn = linear_before_rnn
        self._embeddings_array = []
        self._num_sample_embedding = num_sample_embedding
        self._sample_embedding_file = sample_embedding_file
        self._avgpool_after_conv = avgpool_after_conv
        self._reuse = False
        self._verbose = verbose

        if self._convolutional:
            conv_list = OrderedDict([])
            if self._use_label:
                num_ch = [self._img_size[0] + 1] + [self._num_channels * (2**i) for i in range(self._num_conv)]
            else:
                num_ch = [self._img_size[0]] + [self._num_channels * (2**i) for i in range(self._num_conv)]

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
        else:
            # do not use conv feature extractor
            rnn_input_size = int(self._input_size)

        # whether to use rnn to combine the individual examples' feature
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

        self._embeddings = torch.nn.ModuleList()
        
        for dim in modulation_dims:
            # self._embeddings contains the linear transformation to convert the task embedding to layer wise modulation
            self._embeddings.append(torch.nn.Linear(embedding_input_size, dim))

    def compute_input_size(self, p, k, s, ch):
        current_img_size = self._img_size[1]
        for _ in range(self._num_conv):
            current_img_size = (current_img_size+2*p-k)//s+1
        return ch * int(current_img_size) ** 2

    def forward(self, task, params=None, return_task_embedding=False):
        if not self._reuse and self._verbose: print('='*8 + ' Emb Model ' + '='*8)
        if params is None:
            params = OrderedDict(self.named_parameters())
        
        if self._use_label:
            label_embedding = self._label_representations(task.y).view(-1, 1, self._img_size[1], self._img_size[2])
            x = torch.cat((task.x, label_embedding), dim=1)
        else:
            x = task.x


        if self._convolutional:
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
                x = x.view(x.size(0), -1)
                if not self._reuse and self._verbose: print('flatten: {}'.format(x.size()))
        else:
            # no convolution then just flatten the image as a vector
            x = x.view(x.size(0), -1)
            if not self._reuse and self._verbose: print('flatten: {}'.format(x.size()))

        if self._rnn_aggregation:
            # LSTM input dimensions are seq_len, batch, input_size
            batch_size = 1
            h0 = torch.zeros(self._num_layers*(2 if self._bidirectional else 1),
                             batch_size, self._hidden_size, device=self._device)
            if self._linear_before_rnn: 
                x = F.relu(self.linear(x))
            inputs = x.view(x.size(0), 1, -1)
            output, hn = self.rnn(inputs, h0)
            if self._bidirectional:
                N, B, H = output.shape
                output = output.view(N, B, 2, H // 2)
                embedding_input = torch.cat([output[-1, :, 0], output[0, :, 1]], dim=1)

        else:
            # every example is now represented as a vector
            # average across examples to produce the representation of the dataset
            inputs = F.relu(self.linear(x).view(1, x.size(0), -1).transpose(1, 2))
            if not self._reuse and self._verbose: print('fc: {}'.format(inputs.size()))
            if self._embedding_pooling == 'max':
                # after transpose inputs is of the shape 1, num_features, num_examples
                embedding_input = F.max_pool1d(inputs, x.size(0)).view(1, -1)
            elif self._embedding_pooling == 'avg':
                embedding_input = F.avg_pool1d(inputs, x.size(0)).view(1, -1)
            else:
                raise NotImplementedError
            if not self._reuse and self._verbose: print('reshape after {}pool: {}'.format(
                self._embedding_pooling, embedding_input.size()))

        # randomly sample embedding vectors
        if self._num_sample_embedding != 0:
            self._embeddings_array.append(embedding_input.cpu().clone().detach().numpy())
            if len(self._embeddings_array) >= self._num_sample_embedding:
                if self._sample_embedding_file.split('.')[-1] == 'hdf5':
                    import h5py
                    f = h5py.File(self._sample_embedding_file, 'w')
                    f['embedding'] = np.squeeze(np.stack(self._embeddings_array))
                    f.close()
                elif self._sample_embedding_file.split('.')[-1] == 'pt':
                    torch.save(np.squeeze(np.stack(self._embeddings_array)),
                               self._sample_embedding_file)
                else:
                    raise NotImplementedError

        layer_modulations = []
        # _embeddings 
        for i, embedding in enumerate(self._embeddings):
            embedding_vec = embedding(embedding_input)
            layer_modulations.append(embedding_vec)
            if not self._reuse and self._verbose: print('emb vec {} size: {}'.format(
                i+1, embedding_vec.size()))
        if not self._reuse and self._verbose: print('='*27)
        self._reuse = True

        if return_task_embedding:
            return layer_modulations, embedding_input
        else:
            return layer_modulations

    def to(self, device, **kwargs):
        self._device = device
        super(ConvEmbeddingModel, self).to(device, **kwargs)

