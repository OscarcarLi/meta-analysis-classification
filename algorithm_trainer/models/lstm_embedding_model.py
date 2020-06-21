import torch

class LSTMEmbeddingModel(torch.nn.Module):
    def __init__(self, input_size, output_size, modulation_dims,
                 hidden_size=40, num_layers=2):
        super(LSTMEmbeddingModel, self).__init__()
        self._input_size = input_size
        self._output_size = output_size
        self._modulation_dims = modulation_dims # the dimensions of each layer that we apply modulation on
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._bidirectional = True # always True for bidirectional LSTM
        # self._device = 'cpu'

        rnn_input_size = int(self._input_size + self._output_size)
        self.rnn = torch.nn.LSTM(
            rnn_input_size, hidden_size, num_layers, bidirectional=self._bidirectional)

        self._embeddings = torch.nn.ModuleList()
        for dim in self._modulation_dims:
            self._embeddings.append(torch.nn.Linear(
                hidden_size*(2 if self._bidirectional else 1), dim))

    def forward(self, task, return_task_embedding=False):
        # produce modulations for every single applicable layer
        # batch_size is one because there is only one sequence of training examples
        # all of that specific task
        batch_size = 1
        # h_0, c_0 first hidden state, first cell state
        h0 = torch.zeros(self._num_layers*(2 if self._bidirectional else 1),
                         batch_size, self._hidden_size, device=task.x.device)
        c0 = torch.zeros(self._num_layers*(2 if self._bidirectional else 1),
                         batch_size, self._hidden_size, device=task.x.device)

        x = task.x.view(task.x.size(0), -1)
        y = task.y.view(task.y.size(0), -1)

        '''
        LSTM input dimensions are:
        seq_len=x.size(0) (the number of examples in the training set),
        batch=1,
        input_size=rnn_input_size
        '''
        inputs = torch.cat((x, y), dim=1).view(x.size(0), 1, -1)

        # input of shape (seq_len, batch, input_size): tensor containing the features of the input sequence. The input can also be a packed variable length sequence. See torch.nn.utils.rnn.pack_padded_sequence() or torch.nn.utils.rnn.pack_sequence() for details.
        # h_0 of shape (num_layers * num_directions, batch, hidden_size): tensor containing the initial hidden state for each element in the batch. If the LSTM is bidirectional, num_directions should be 2, else it should be 1.
        # c_0 of shape (num_layers * num_directions, batch, hidden_size): tensor containing the initial cell state for each element in the batch.
        # If (h_0, c_0) is not provided, both h_0 and c_0 default to zero.
        output, (hn, cn) = self.rnn(inputs, (h0, c0))
        '''
        output of shape (seq_len, batch, num_directions * hidden_size): tensor 
        containing the output features (h_t) from the last layer of the LSTM, for each t.
        '''
        if self._bidirectional:
        # N is the size of the training set
        # B = 1
        # H has 2 * hidden_size because we are using bi-directional
            N, B, H = output.shape             
            output = output.view(N, B, 2, H // 2)
            task_embedding = torch.cat(
                [output[-1, :, 0, :], output[0, :, 1, :]], dim=1) # : takes the mid two dimensions
            assert task_embedding.size(0) == 1
            assert task_embedding.size(1) == 2 * self._hidden_size
        # embedding input is the task embedding 
        # construct layer wise modulation parameter
        layer_modulations = []
        for embedding in self._embeddings:
            # embedding is a linear transformation to take the embedding_input (task embedding)
            # and maps it to layerwise modulation
            layer_modulations.append(embedding(task_embedding))
        if return_task_embedding:
            return layer_modulations, task_embedding
        else:
            return layer_modulations

    # def to(self, device, **kwargs):
    #     self._device = device
    #     super(LSTMEmbeddingModel, self).to(device, **kwargs)



class LSTMAttentionEmbeddingModel(torch.nn.Module):
    def __init__(self, input_size, output_size, modulation_dims,
                 hidden_size=40, num_layers=2, gamma=2., basis_size=3, gamma_period=300):
        super(LSTMAttentionEmbeddingModel, self).__init__()
        self._input_size = input_size
        self._output_size = output_size
        self._modulation_dims = modulation_dims # the dimensions of each layer that we apply modulation on
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._bidirectional = True # always True for bidirectional LSTM
        # self._device = 'cpu'

        rnn_input_size = int(self._input_size + self._output_size)
        self.rnn = torch.nn.LSTM(
            rnn_input_size, hidden_size, num_layers, bidirectional=self._bidirectional)

        self._basis_size = basis_size 
        self._gamma_period = gamma_period
        self._embedding_matrices = torch.nn.ParameterList()
        for dim in self._modulation_dims:
            self._embedding_matrices.append(
                torch.nn.Parameter(torch.randn(self._basis_size, dim), requires_grad=True)
            )
        self._attn = torch.nn.ModuleList()
        for _ in range(len(self._modulation_dims)):
            self._attn.append(torch.nn.Linear(
                hidden_size*(2 if self._bidirectional else 1), self._basis_size))

        self._softmax_layer = torch.nn.Softmax(dim=-1)
        self._gamma= gamma

    def forward(self, task, return_task_embedding=False, iter=None):
        # produce modulations for every single applicable layer
        # batch_size is one because there is only one sequence of training examples
        # all of that specific task
        if iter is not None and iter % self._gamma_period  == 0:
            self._gamma += 1
            self._gamma_period *=2 
            print(f"Setting value of gamma to : {self._gamma} and gamma period to {self._gamma_period}")

        batch_size = 1
        # h_0, c_0 first hidden state, first cell state
        h0 = torch.zeros(self._num_layers*(2 if self._bidirectional else 1),
                         batch_size, self._hidden_size, device=task.x.device)
        c0 = torch.zeros(self._num_layers*(2 if self._bidirectional else 1),
                         batch_size, self._hidden_size, device=task.x.device)

        x = task.x.view(task.x.size(0), -1)
        y = task.y.view(task.y.size(0), -1)

        '''
        LSTM input dimensions are:
        seq_len=x.size(0) (the number of examples in the training set),
        batch=1,
        input_size=rnn_input_size
        '''
        inputs = torch.cat((x, y), dim=1).view(x.size(0), 1, -1)

        # input of shape (seq_len, batch, input_size): tensor containing the features of the input sequence. The input can also be a packed variable length sequence. See torch.nn.utils.rnn.pack_padded_sequence() or torch.nn.utils.rnn.pack_sequence() for details.
        # h_0 of shape (num_layers * num_directions, batch, hidden_size): tensor containing the initial hidden state for each element in the batch. If the LSTM is bidirectional, num_directions should be 2, else it should be 1.
        # c_0 of shape (num_layers * num_directions, batch, hidden_size): tensor containing the initial cell state for each element in the batch.
        # If (h_0, c_0) is not provided, both h_0 and c_0 default to zero.
        output, (hn, cn) = self.rnn(inputs, (h0, c0))
        '''
        output of shape (seq_len, batch, num_directions * hidden_size): tensor 
        containing the output features (h_t) from the last layer of the LSTM, for each t.
        '''
        if self._bidirectional:
        # N is the size of the training set
        # B = 1
        # H has 2 * hidden_size because we are using bi-directional
            N, B, H = output.shape             
            output = output.view(N, B, 2, H // 2)
            task_embedding = torch.cat(
                [output[-1, :, 0, :], output[0, :, 1, :]], dim=1) # : takes the mid two dimensions
            assert task_embedding.size(0) == 1
            assert task_embedding.size(1) == 2 * self._hidden_size
        # embedding input is the task embedding 
        # construct layer wise modulation parameter
        layer_modulations = []
        for attn_layer, emb_matrix in zip(self._attn, self._embedding_matrices):
            # attn_layer is a linear transformation to take the embedding_input (task embedding)
            # and maps it to attn wts. These are then used for basis selection
            # where basis is given by embedding matrices
            wts = attn_layer(task_embedding)
            wts = self.get_skewed_wts(wts) # normalization to convex combination coefficients
            layer_modulations.append(torch.mm(wts, emb_matrix))
        if return_task_embedding:
            return layer_modulations, task_embedding
        else:
            return layer_modulations

    def get_skewed_wts(self, wts):
        wts = wts * self._gamma
        wts = self._softmax_layer(wts)
        return wts

    # def to(self, device, **kwargs):
    #     self._device = device
    #     super(LSTMEmbeddingModel, self).to(device, **kwargs)



class RegLSTMEmbeddingModel(torch.nn.Module):
    def __init__(self, input_size, output_size, modulation_mat_size,
                 hidden_size=40, num_layers=2):
        super(RegLSTMEmbeddingModel, self).__init__()
        self._input_size = input_size
        self._output_size = output_size
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._bidirectional = True # always True for bidirectional LSTM
        # self._device = 'cpu'

        rnn_input_size = int(self._input_size + self._output_size)
        self.rnn = torch.nn.LSTM(
            rnn_input_size, hidden_size, num_layers, bidirectional=self._bidirectional)

        # modulation_mat_size a tuple (low-rank size, feature_space_dimension)
        self.modulation_mat_generator = torch.nn.ModuleList()
        for _ in range(modulation_mat_size[0]):
            # bias=True as default
            self.modulation_mat_generator.append(torch.nn.Linear(
                hidden_size*(2 if self._bidirectional else 1), modulation_mat_size[1]
            ))

        self.modulation_bias_generator = torch.nn.Linear(
            hidden_size*(2 if self._bidirectional else 1), modulation_mat_size[0]
        )

    def forward(self, task, return_task_embedding=False):
        # produce modulations for every single applicable layer
        # batch_size is one because there is only one sequence of training examples
        # all of that specific task
        batch_size = 1
        # h_0, c_0 first hidden state, first cell state
        h0 = torch.zeros(self._num_layers*(2 if self._bidirectional else 1),
                         batch_size, self._hidden_size, device=task.x.device)
        c0 = torch.zeros(self._num_layers*(2 if self._bidirectional else 1),
                         batch_size, self._hidden_size, device=task.x.device)

        x = task.x.view(task.x.size(0), -1)
        y = task.y.view(task.y.size(0), -1)

        '''
        LSTM input dimensions are:
        seq_len=x.size(0) (the number of examples in the training set),
        batch=1,
        input_size=rnn_input_size
        '''
        inputs = torch.cat((x, y), dim=1).view(x.size(0), 1, -1)

        # input of shape (seq_len, batch, input_size): tensor containing the features of the input sequence. The input can also be a packed variable length sequence. See torch.nn.utils.rnn.pack_padded_sequence() or torch.nn.utils.rnn.pack_sequence() for details.
        # h_0 of shape (num_layers * num_directions, batch, hidden_size): tensor containing the initial hidden state for each element in the batch. If the LSTM is bidirectional, num_directions should be 2, else it should be 1.
        # c_0 of shape (num_layers * num_directions, batch, hidden_size): tensor containing the initial cell state for each element in the batch.
        # If (h_0, c_0) is not provided, both h_0 and c_0 default to zero.
        output, (hn, cn) = self.rnn(inputs, (h0, c0))
        '''
        output of shape (seq_len, batch, num_directions * hidden_size): tensor 
        containing the output features (h_t) from the last layer of the LSTM, for each t.
        '''
        if self._bidirectional:
        # N is the size of the training set
        # B = 1
        # H has 2 * hidden_size because we are using bi-directional
            N, B, H = output.shape             
            output = output.view(N, B, 2, H // 2)
            task_embedding = torch.cat(
                [output[-1, :, 0, :], output[0, :, 1, :]], dim=1) # : takes the mid two dimensions
            assert task_embedding.size(0) == 1
            assert task_embedding.size(1) == 2 * self._hidden_size
        # embedding input is the task embedding 
        # construct layer wise modulation parameter
        
        modulation_mat = []
        for mod in self.modulation_mat_generator:
            modulation_mat.append(
                mod(task_embedding)
            )

        modulation_mat = torch.cat(modulation_mat, dim=0)

        modulation_bias = self.modulation_bias_generator(task_embedding)

        if return_task_embedding:
            return (modulation_mat, modulation_bias), task_embedding
        else:
            return (modulation_mat, modulation_bias)