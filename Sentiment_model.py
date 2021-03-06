from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LSTMModel(nn.Module):
    """
    LSTMModel(
      (embedding): Embedding(54848, 50)
      (encoder): LSTM(50, 100, num_layers=2, dropout=0.2, bidirectional=True)
      (decoder1): Linear(in_features=400, out_features=100, bias=True)
      (decoder2): Linear(in_features=100, out_features=2, bias=True)
    )

    vocab_size:词汇量
    embedding_dim:词向量维度
    pretrained_weight:预训练权重
    update_w2v:是否在训练中更新w2v
    hidden_dim:隐藏层节点数
    num_layers:LSTM层数
    drop_keep_prob:# dropout层，参数keep的比例
    n_class:分类数，分别为pos和neg
    bidirectional:是否使用双向LSTM
    """
    def __init__(self, vocab_size, embedding_dim, pretrained_weight, update_w2v, hidden_dim,
                 num_layers, drop_keep_prob, n_class, bidirectional, **kwargs):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.n_class = n_class
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding.from_pretrained(pretrained_weight)
        self.embedding.weight.requires_grad = update_w2v
        self.encoder = nn.LSTM(input_size=embedding_dim, hidden_size=self.hidden_dim,
                               num_layers=num_layers, bidirectional=self.bidirectional,
                               dropout=drop_keep_prob)
        # 根据是否使用双向LSTM，初始化Linear层（decoder层）
        if self.bidirectional:
            self.decoder1 = nn.Linear(hidden_dim * 4, hidden_dim)
            self.decoder2 = nn.Linear(hidden_dim, n_class)
        else:
            self.decoder1 = nn.Linear(hidden_dim * 2, hidden_dim)
            self.decoder2 = nn.Linear(hidden_dim, n_class)

    def forward(self, inputs):
        embeddings = self.embedding(inputs)  # [batch, seq_len][64, 65] => [batch, seq_len, embed_dim][64,65,50]
        embeddings_permute = embeddings.permute([1, 0, 2])  # [seq_len, batch, embed_dim][65, 64, 50]
        states, hidden = self.encoder(embeddings_permute)
        encoding = torch.cat([states[0], states[-1]], dim=1)

        # print('inputs size,', np.shape(inputs))  # [64, 65]
        # print('embeddings size,', np.shape(embeddings))  # [64, 65, 50]
        # print('embeddings_permute size,', np.shape(embeddings_permute))  # [65, 64, 50]
        # print('states size,', np.shape(states))  # [65, 64, 200]
        # print('hidden size,', np.shape(hidden))  # [4, 64, 100] [4, 64, 100]
        # print('encoding size,', np.shape(encoding))  # [64, 400]

        outputs = self.decoder1(encoding)
        # print('outputs size,', np.shape(outputs))  # [64, 100]
        # outputs = F.softmax(outputs, dim=1)
        outputs = self.decoder2(outputs)
        # print('outputs size,', np.shape(outputs))  # [64, 2]
        return outputs


class LSTM_attention(nn.Module):
    """
    LSTM_attention(
      (embedding): Embedding(54848, 50)
      (encoder): LSTM(50, 100, num_layers=2, dropout=0.2, bidirectional=True)
      (decoder1): Linear(in_features=200, out_features=100, bias=True)
      (decoder2): Linear(in_features=100, out_features=2, bias=True)
    )
    """
    def __init__(self, vocab_size, embedding_dim, pretrained_weight, update_w2v, hidden_dim,
                 num_layers, drop_keep_prob, n_class, bidirectional, **kwargs):
        super(LSTM_attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.n_class = n_class

        self.bidirectional = bidirectional
        self.embedding = nn.Embedding.from_pretrained(pretrained_weight)
        self.embedding.weight.requires_grad = update_w2v
        self.encoder = nn.LSTM(input_size=embedding_dim, hidden_size=self.hidden_dim,
                               num_layers=num_layers, bidirectional=self.bidirectional,
                               dropout=drop_keep_prob)

        # TODO
        # What is nn. Parameter ? Explain
        self.weight_W = nn.Parameter(torch.Tensor(2 * hidden_dim, 2 * hidden_dim))
        self.weight_proj = nn.Parameter(torch.Tensor(2 * hidden_dim, 1))

        if self.bidirectional:
            # self.decoder1 = nn.Linear(hidden_dim * 2, n_class)
            self.decoder1 = nn.Linear(hidden_dim * 2, hidden_dim)
            self.decoder2 = nn.Linear(hidden_dim, n_class)
        else:
            self.decoder1 = nn.Linear(hidden_dim * 2, hidden_dim)
            self.decoder2 = nn.Linear(hidden_dim, n_class)

        nn.init.uniform_(self.weight_W, -0.1, 0.1)
        nn.init.uniform_(self.weight_proj, -0.1, 0.1)

    def forward(self, inputs):
        embeddings = self.embedding(inputs)  # [batch, seq_len] => [batch, seq_len, embed_dim][64,75,50]

        states, hidden = self.encoder(embeddings.permute([0, 1, 2]))  # [batch, seq_len, embed_dim]
        # attention

        u = torch.tanh(torch.matmul(states, self.weight_W))
        att = torch.matmul(u, self.weight_proj)

        att_score = F.softmax(att, dim=1)
        scored_x = states * att_score
        encoding = torch.sum(scored_x, dim=1)
        outputs = self.decoder1(encoding)
        outputs = self.decoder2(outputs)
        return outputs
