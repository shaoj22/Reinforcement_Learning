# network model
# author: Charles Lee
# date: 2022.10.26

import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn
from graph_encoder import *

class Attention_Net(nn.Module):
    def __init__(self, node_dim):
        super().__init__()
        embed_dim = 128
        self.encoder = GraphAttentionEncoder(
            n_heads=8, 
            embed_dim=embed_dim, 
            n_layers=2, 
            node_dim=node_dim, 
            normalization="batch", 
            feed_forward_hidden=128
        )
        self.decoder = nn.Linear(embed_dim, 2)

    def forward(self, state):
        if not isinstance(state, torch.FloatTensor):
            state = torch.FloatTensor(state)
        embeddings, _ = self.encoder(state)
        logits = self.decoder(embeddings)
        return logits

class AttentionQuery_Net(nn.Module):
    def __init__(self, node_dim):
        super().__init__()
        embed_dim = 128
        self.encoder = GraphAttentionEncoder(
            n_heads=2, 
            embed_dim=embed_dim, 
            n_layers=2, 
            node_dim=node_dim, 
            normalization="batch", 
            feed_forward_hidden=128
        )
        self.decoder = MultiHeadAttentionMatch(
            n_heads=1, 
            input_dim=embed_dim, 
            embed_dim=embed_dim,
        )

    def forward(self, state, q):
        if not isinstance(state, torch.FloatTensor):
            state = torch.FloatTensor(state)
        state = state[None]
        embeddings, _ = self.encoder(state)
        query = embeddings[:, q:q+1, :]
        logits = self.decoder(query, embeddings)[0][0]
        return logits

class MultiHeadAttentionMatch(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadAttentionMatch, self).__init__()

        if val_dim is None:
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        self.W_out = nn.Parameter(torch.Tensor(n_heads, val_dim, embed_dim))

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h, mask=None):
        """

        :param q: queries (batch_size, n_query, input_dim)
        :param h: data (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """
        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        # Optionally apply mask to prevent attention
        if mask is not None:
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            compatibility[mask] = -np.inf

        return compatibility[0]


