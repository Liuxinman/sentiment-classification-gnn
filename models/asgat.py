# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.dynamic_rnn import DynamicLSTM


class GraphAttention(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, n_heads = 8,bias=True):
        super(GraphAttention, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.is_concat = True
        self.n_heads = n_heads
        self.alpha = 0.2
        self.dropout = 0.6
        if self.is_concat:
            assert out_features % self.n_heads == 0
            self.n_hidden = out_features // self.n_heads

        # (1) self.W: Linear layer that transform the input feature before self attention.
        # You should NOT use for loops for the multiheaded implementation (set bias = Flase)
        self.W = nn.Linear(in_features, self.n_hidden * self.n_heads, bias=False)
        # (2) self.attention: Linear layer that compute the attention score (set bias = Flase)
        self.attention = nn.Linear(2 * self.n_hidden, 1, bias=False)
        # (3) self.activation: Activation function (LeakyReLU whith negative_slope=alpha)
        self.activation = nn.LeakyReLU(negative_slope=self.alpha)
        # (4) self.softmax: Softmax function (what's the dim to compute the summation?)
        self.softmax = nn.Softmax(dim=1)
        # (5) self.dropout_layer: Dropout function(with ratio=dropout)
        self.dropout_layer = nn.Dropout(p=self.dropout)

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, h, adj):
        # Number of nodes
        batch_size = h.shape[0]
        n_nodes = h.shape[1]
        # (1) calculate s = Wh and reshape it to [n_nodes, n_heads, n_hidden]
        s = self.W(h).view(batch_size,n_nodes,self.n_heads, self.n_hidden)

        # (2) get [s_i || s_j] using tensor.repeat(), repeat_interleave(), torch.cat(), tensor.view()
        s_i = s.repeat_interleave(s.shape[1], dim=1).view(batch_size,n_nodes,n_nodes,self.n_heads,self.n_hidden)
        s_j = s.unsqueeze(1).repeat(1,s.shape[1],1,1,1)
        e = torch.cat((s_i, s_j), dim=-1)
        # (3) apply the attention layer
        e = self.attention(e)

        # (4) apply the activation layer (you will get the attention score e)
        e = self.activation(e)

        # (5) remove the last dimension 1 use tensor.squeeze()
        e = e.squeeze(-1)

        # (6) mask the attention score with the adjacency matrix (if there's no edge, assign it to -inf)
        masked_e = e.masked_fill(adj.unsqueeze(-1) == 0,-1e10)
        # (7) apply softmax
        a = self.softmax(masked_e)
        # (8) apply dropout_layer
        a = self.dropout_layer(a)

        h_prime = torch.einsum('bijh,bjhf->bihf', a, s) #[n_nodes, n_heads, n_hidden]


        #print(h_prime)
        if self.is_concat:
            ############## Your code here #########################################
            output = h_prime.reshape(batch_size,n_nodes,-1)
            #######################################################################
        # Take the mean of the heads (for the last layer)
        else:
            ############## Your code here #########################################
            output = h_prime.mean(dim=2)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class ASGAT(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(ASGAT, self).__init__()
        self.opt = opt
        self.n_heads = 8
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.text_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.gat1 = GraphAttention(2*opt.hidden_dim, 2*opt.hidden_dim,self.n_heads)
        self.gat2 = GraphAttention(2*opt.hidden_dim, 2*opt.hidden_dim,n_heads = 1)
        self.fc = nn.Linear(2*opt.hidden_dim, opt.polarities_dim)
        self.text_embed_dropout = nn.Dropout(0.6)

    def position_weight(self, x, aspect_double_idx, text_len, aspect_len):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        text_len = text_len.cpu().numpy()
        aspect_len = aspect_len.cpu().numpy()
        weight = [[] for i in range(batch_size)]
        for i in range(batch_size):
            context_len = text_len[i] - aspect_len[i]
            for j in range(aspect_double_idx[i,0]):
                weight[i].append(1-(aspect_double_idx[i,0]-j)/context_len)
            for j in range(aspect_double_idx[i,0], aspect_double_idx[i,1]+1):
                weight[i].append(0)
            for j in range(aspect_double_idx[i,1]+1, text_len[i]):
                weight[i].append(1-(j-aspect_double_idx[i,1])/context_len)
            for j in range(text_len[i], seq_len):
                weight[i].append(0)
        weight = torch.tensor(weight, dtype=torch.float).unsqueeze(2).to(self.opt.device)
        return weight*x

    def mask(self, x, aspect_double_idx):
        batch_size, seq_len = x.shape[0], x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        mask = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for j in range(aspect_double_idx[i,0]):
                mask[i].append(0)
            for j in range(aspect_double_idx[i,0], aspect_double_idx[i,1]+1):
                mask[i].append(1)
            for j in range(aspect_double_idx[i,1]+1, seq_len):
                mask[i].append(0)
        mask = torch.tensor(mask, dtype=torch.float).unsqueeze(2).to(self.opt.device)
        return mask*x

    def forward(self, inputs):
        text_indices, aspect_indices, left_indices, adj = inputs
        text_len = torch.sum(text_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        left_len = torch.sum(left_indices != 0, dim=-1)
        aspect_double_idx = torch.cat([left_len.unsqueeze(1), (left_len+aspect_len-1).unsqueeze(1)], dim=1)
        text = self.embed(text_indices)
        text = self.text_embed_dropout(text)
        text_out, (_, _) = self.text_lstm(text, text_len)
        x = F.relu(self.gat1(self.position_weight(text_out, aspect_double_idx, text_len, aspect_len), adj))
        x = F.relu(self.gat2(self.position_weight(x, aspect_double_idx, text_len, aspect_len), adj))
        x = self.mask(x, aspect_double_idx)
        alpha_mat = torch.matmul(x, text_out.transpose(1, 2))
        alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        x = torch.matmul(alpha, text_out).squeeze(1) # batch_size x 2*hidden_dim
        output = self.fc(x)
        return output