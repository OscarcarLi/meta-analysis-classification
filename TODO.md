Things to read
- models
files checked
    - fully_connected.py (use regression)
    - gated_net.py (use regression)
    - lstm_embedding_model.py (use regression)
    - gru_embedding_model.py (not used in regression)
    - conv_net.py (fixed a bug in the original code base)
    - gated_conv_net.py
    - conv_embedding_model.py (use convnet to extract the features and then use aggregation either thru rnn or averaging/max to get a task embedding, then produce layerwise modulation)
- dataset
    - where is the mnist dataset
    - understand multimodal few shot dataset how it's sampling the tasks with mix_mini and mix_meta
    - cifar forgot to normalize (fixed)
    - total batches before actually means total number of tasks

- currently the ConvEmbeddingModel is not taking in information about the labels, how to incorporate this?
- how should we do batch norm during test time?
- should we use rnn to produce mmaml's layer modulations
- implement RegConvEmbeddingModel
- implement RegMAML