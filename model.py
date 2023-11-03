import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNModel(nn.Module):
    """带有编码器、循环模块和解码器的容器模块"""

    def __init__(self, rnn_type, vocab_size, embedding_dim, hidden_dim, num_of_layers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.vocab_size = vocab_size
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(vocab_size, embedding_dim)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(embedding_dim, hidden_dim, num_of_layers,
                                             dropout=dropout)  # 使用getattr()返回一个类，并赋值
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]  # 进到此else说明是普通RNN，用rnn_type决定激活函数
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(embedding_dim, hidden_dim, num_of_layers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(hidden_dim, vocab_size)

        if tie_weights:
            if hidden_dim != embedding_dim:
                raise ValueError('When using the tied flag, hidden_dim must be equal to embedding_size')
            self.decoder.weight = self.encoder.weight

        self.init_weights()
        self.rnn_type = rnn_type
        self.hidden_dim = hidden_dim
        self.num_of_layers = num_of_layers

    def init_weights(self):
        init_range = 0.1
        nn.init.uniform_(self.encoder.weight, -init_range, init_range)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -init_range, init_range)

    def forward(self, input, hidden):
        # self.inspect_forward(input, hidden)
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.vocab_size)
        return F.log_softmax(decoded, dim=1), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.num_of_layers, bsz, self.hidden_dim),
                    weight.new_zeros(self.num_of_layers, bsz, self.hidden_dim))
        else:
            return weight.new_zeros(self.num_of_layers, bsz, self.hidden_dim)

    def inspect_forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.vocab_size)
        return F.log_softmax(decoded, dim=1), hidden

    def get_word_sim(self, w1, w2):
        """
        返回 w1 和 w2 之间的余弦相似度
        """
        e1 = self.encoder(w1)
        e2 = self.encoder(w2)
        return F.cosine_similarity(e1, e2, dim=0)