#!usr/bin/env python
#-*- coding:utf-8 -*-

import torch
import torch.nn as nn

class BiGRU(nn.Module):
    """
    decotor 使用 BiGRU
    """
    def __init__(self, embedding_size, hidden, n_layers, dropout=0.0):
        super(BiGRU, self).__init__()
        self.rnn = nn.GRU(embedding_size, hidden, num_layers=n_layers,
                          bidirectional=True, dropout=dropout, batch_first=True)
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(hidden*2, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        prob = self.sigmoid(self.linear(out))
        return prob


if __name__ == "__main__":
    model = BiGRU(2, 2, 2)
    text = torch.Tensor([[[1,1],[2,2],[3,3],[4,4]]])
    p = model(text)
    print(p)
    print(text * p)
    print(1-p)
