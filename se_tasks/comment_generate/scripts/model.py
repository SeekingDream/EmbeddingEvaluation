# -*- coding: utf-8 -*-

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class EncRNN(nn.Module):
    def __init__(self, vsz, embed_dim, hidden_dim, n_layers, use_birnn, dout, embed_vec, padding_index=1):
        super(EncRNN, self).__init__()
        if torch.is_tensor(embed_vec):
            self.embed = nn.Embedding(vsz, embed_dim, padding_idx=padding_index, _weight=embed_vec)
            self.embed.weight.requires_grad = False
        else:
            self.embed = nn.Embedding(vsz, embed_dim, padding_idx=padding_index)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, n_layers,
                           bidirectional=use_birnn, batch_first=True)
        self.dropout = nn.Dropout(dout)

    def forward(self, inputs, seq_lengths):
        embs = self.dropout(self.embed(inputs))

        packed_input = pack_padded_sequence(embs, seq_lengths.cpu().numpy(), batch_first=True)
        enc_outs, hidden = self.rnn(packed_input)
        enc_outs, _ = pad_packed_sequence(enc_outs, batch_first=True)

        return self.dropout(enc_outs), hidden


class Attention(nn.Module):
    def __init__(self, hidden_dim, method):
        super(Attention, self).__init__()
        self.method = method
        self.hidden_dim = hidden_dim

        if method == 'general':
            self.w = nn.Linear(hidden_dim, hidden_dim)
        elif method == 'concat':
            self.w = nn.Linear(hidden_dim * 2, hidden_dim)
            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_dim))

    def forward(self, dec_out, enc_outs):
        if self.method == 'dot':
            attn_energies = self.dot(dec_out, enc_outs)
        elif self.method == 'general':
            attn_energies = self.general(dec_out, enc_outs)
        elif self.method == 'concat':
            attn_energies = self.concat(dec_out, enc_outs)
        return F.softmax(attn_energies, dim=0)

    def dot(self, dec_out, enc_outs):
        return torch.sum(dec_out * enc_outs, dim=2)

    def general(self, dec_out, enc_outs):
        energy = self.w(enc_outs)
        return torch.sum(dec_out * energy, dim=2)

    def concat(self, dec_out, enc_outs):
        dec_out = dec_out.expand(enc_outs.shape[0], -1, -1)
        energy = torch.cat((dec_out, enc_outs), 2)
        return torch.sum(self.v * self.w(energy).tanh(), dim=2)


class DecRNN(nn.Module):
    def __init__(self, vsz, embed_dim, hidden_dim, n_layers, use_birnn,
                 dout, attn, tied, padding_index=1):
        super(DecRNN, self).__init__()
        hidden_dim = hidden_dim * 2 if use_birnn else hidden_dim

        self.embed = nn.Embedding(vsz, embed_dim, padding_idx=padding_index)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, n_layers, batch_first=True)

        self.w = nn.Linear(hidden_dim * 2, hidden_dim)
        self.attn = Attention(hidden_dim, attn)

        self.out_projection = nn.Linear(hidden_dim, vsz)
        if tied:
            if embed_dim != hidden_dim:
                raise ValueError(
                    f"when using the tied flag, embed-dim:{embed_dim} \
                    must be equal to hidden-dim:{hidden_dim}")
            self.out_projection.weight = self.embed.weight
        self.dropout = nn.Dropout(dout)

    def forward(self, inputs, hidden, enc_outs, seq_lengths):
        inputs = inputs.unsqueeze(0)
        embs = self.dropout(self.embed(inputs))

        packed_input = pack_padded_sequence(embs, seq_lengths.cpu().numpy(), batch_first=True)
        dec_out, hidden = self.rnn(packed_input, hidden)
        dec_out, _ = pad_packed_sequence(dec_out, batch_first=True)

        attn_weights = self.attn(dec_out, enc_outs).transpose(1, 0)
        enc_outs = enc_outs.transpose(1, 0)
        context = torch.bmm(attn_weights.unsqueeze(1), enc_outs)
        cats = self.w(torch.cat((dec_out, context.transpose(1, 0)), dim=2))
        pred = self.out_projection(cats.tanh().squeeze(0))
        return pred, hidden


class Seq2seqAttn(nn.Module):
    def __init__(self, args, src_vsz, tgt_vsz, device, embed_vec):
        super().__init__()
        self.src_vsz = src_vsz
        self.tgt_vsz = tgt_vsz
        self.encoder = EncRNN(self.src_vsz, args.embed_dim, args.hidden_dim,
                              args.n_layers, args.bidirectional, args.dropout, embed_vec)
        self.decoder = DecRNN(self.tgt_vsz, args.embed_dim, args.hidden_dim,
                              args.n_layers, args.bidirectional, args.dropout,
                              args.attn, args.tied)
        self.device = device
        self.n_layers = args.n_layers
        self.hidden_dim = args.hidden_dim
        self.use_birnn = args.bidirectional

    def forward(self, srcs, src_len, tgts=None, tar_len=None, maxlen=100, tf_ratio=0.0):
        bsz, slen = srcs.size()
        tlen = tgts.size(1) if isinstance(tgts, torch.Tensor) else maxlen
        tf_ratio = tf_ratio if isinstance(tgts, torch.Tensor) else 0.0

        enc_outs, hidden = self.encoder(srcs, src_len)

        dec_inputs = torch.ones_like(srcs[1]) * 2  # <eos> is mapped to id=2
        outs = []

        if self.use_birnn:
            def trans_hidden(hs):
                hs = hs.view(self.n_layers, 2, bsz, self.hidden_dim)
                hs = torch.stack([torch.cat((h[0], h[1]), 1) for h in hs])
                return hs

            hidden = tuple(trans_hidden(hs) for hs in hidden)

        for i in range(tlen):
            preds, hidden = self.decoder(dec_inputs, hidden, enc_outs, tar_len)
            outs.append(preds)
            use_tf = random.random() < tf_ratio
            dec_inputs = tgts[i] if use_tf else preds.max(1)[1]
        return torch.stack(outs)
