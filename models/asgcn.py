# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.dynamic_rnn import DynamicLSTM
from transformers import BertModel


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, text, adj):
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj, hidden) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class BertEmbedding(nn.Module):
    def __init__(self, emb_dim, bert_version, device) -> None:
        super(BertEmbedding, self).__init__()
        self.bert_model = BertModel.from_pretrained(bert_version)
        self.linear = nn.Linear(self.bert_model.config.hidden_size, emb_dim)
        self.device = device

    def forward(self, input_ids, token_type_ids, attention_mask, word_ids, token_lengths):
        output = self.bert_model(input_ids, token_type_ids, attention_mask)
        sentence_hidden_states = output.last_hidden_state
        batch_size, _, emb_size = sentence_hidden_states.shape
        all_token_embeddings = torch.zeros(batch_size, max(token_lengths), emb_size)
        for i in torch.arange(batch_size):
            for _id in torch.arange(token_lengths[i]):
                all_token_embeddings[i, _id, :] = torch.nan_to_num(
                    sentence_hidden_states[i][word_ids[i] == _id].mean(dim=0)
                )
        all_token_embeddings = all_token_embeddings.to(self.device)
        all_token_embeddings = self.linear(all_token_embeddings)
        return all_token_embeddings


class ASGCN(nn.Module):
    def __init__(self, embedding_matrix=None, opt=None):
        super(ASGCN, self).__init__()
        self.opt = opt
        self.use_bert = opt.use_bert
        if self.use_bert:
            # self.embed = BertEmbedding(2 * opt.hidden_dim, self.opt.bert_version, self.opt.device)
            self.embed = BertEmbedding(opt.embed_dim, self.opt.bert_version, self.opt.device)
        else:
            self.embed = nn.Embedding.from_pretrained(
                torch.tensor(embedding_matrix, dtype=torch.float)
            )
        self.text_lstm = DynamicLSTM(
            opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True
        )
        self.gc1 = GraphConvolution(2 * opt.hidden_dim, 2 * opt.hidden_dim)
        self.gc2 = GraphConvolution(2 * opt.hidden_dim, 2 * opt.hidden_dim)
        self.fc = nn.Linear(2 * opt.hidden_dim, opt.polarities_dim)
        self.text_embed_dropout = nn.Dropout(0.3)

    def position_weight(self, x, aspect_double_idx, text_len, aspect_len):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        text_len = text_len.cpu().numpy()
        aspect_len = aspect_len.cpu().numpy()
        weight = [[] for i in range(batch_size)]
        for i in range(batch_size):
            context_len = text_len[i] - aspect_len[i]
            for j in range(aspect_double_idx[i, 0]):
                weight[i].append(1 - (aspect_double_idx[i, 0] - j) / context_len)
            for j in range(aspect_double_idx[i, 0], aspect_double_idx[i, 1] + 1):
                weight[i].append(0)
            for j in range(aspect_double_idx[i, 1] + 1, text_len[i]):
                weight[i].append(1 - (j - aspect_double_idx[i, 1]) / context_len)
            for j in range(text_len[i], seq_len):
                weight[i].append(0)
        weight = torch.tensor(weight, dtype=torch.float).unsqueeze(2).to(self.opt.device)
        return weight * x

    def mask(self, x, aspect_double_idx):
        batch_size, seq_len = x.shape[0], x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        mask = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for j in range(aspect_double_idx[i, 0]):
                mask[i].append(0)
            for j in range(aspect_double_idx[i, 0], aspect_double_idx[i, 1] + 1):
                mask[i].append(1)
            for j in range(aspect_double_idx[i, 1] + 1, seq_len):
                mask[i].append(0)
        mask = torch.tensor(mask, dtype=torch.float).unsqueeze(2).to(self.opt.device)
        return mask * x

    def forward(self, inputs):
        (
            text_indices,
            aspect_indices,
            left_indices,
            adj,
            token_type_ids,
            attention_mask,
            word_ids,
            text_indices_bert,
        ) = inputs
        text_len = torch.sum(text_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        left_len = torch.sum(left_indices != 0, dim=-1)
        aspect_double_idx = torch.cat(
            [left_len.unsqueeze(1), (left_len + aspect_len - 1).unsqueeze(1)], dim=1
        )
        if self.use_bert:
            text = self.embed(text_indices_bert, token_type_ids, attention_mask, word_ids, text_len)
            text = self.text_embed_dropout(text)
        else:
            text = self.embed(text_indices)
            text = self.text_embed_dropout(text)
        text_out, (_, _) = self.text_lstm(text, text_len)
        assert text.shape[1] == max(text_len), "emb seq len does not match text len"
        x = F.relu(
            self.gc1(self.position_weight(text_out, aspect_double_idx, text_len, aspect_len), adj)
        )
        x = F.relu(self.gc2(self.position_weight(x, aspect_double_idx, text_len, aspect_len), adj))
        x = self.mask(x, aspect_double_idx)
        alpha_mat = torch.matmul(x, text_out.transpose(1, 2))
        alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        x = torch.matmul(alpha, text_out).squeeze(1)  # batch_size x 2*hidden_dim
        output = self.fc(x)
        return output
