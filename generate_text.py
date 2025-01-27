#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math

D_MODEL = 512
N_HEAD = 8
N_LAYERS = 6
D_FF = 2048
DROPOUT = 0.1
MAX_SEQ_LEN = 100
VOCAB_SIZE = 1000


class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


def scaled_dot_product_attention(query, key, value, mask=None):
    d_k = query.size(-1)
    attn_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
    attn_probs = F.softmax(attn_scores, dim=-1)
    output = torch.matmul(attn_probs, value)
    return output, attn_probs


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        q = self.W_q(query).view(batch_size,
                                 -1, self.n_head, self.d_k).transpose(1, 2)
        k = self.W_k(key).view(batch_size,
                               -1, self.n_head, self.d_k).transpose(1, 2)
        v = self.W_v(value).view(batch_size,
                                 -1, self.n_head, self.d_k).transpose(1, 2)

        attn_output, attn_probs = scaled_dot_product_attention(q, k, v, mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
                batch_size, -1, self.d_model)

        output = self.W_o(attn_output)
        return output, attn_probs


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(d_model, n_head)
        self.ffn = FeedForwardNetwork(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output, _ = self.attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head)
        self.enc_attn = MultiHeadAttention(d_model, n_head)
        self.ffn = FeedForwardNetwork(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        enc_attn_output, _ = self.enc_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(enc_attn_output))
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_output))
        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_head,
                 n_layers, d_ff, dropout, max_seq_len):
        super(Encoder, self).__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        self.layers = nn.ModuleList(
                [EncoderLayer(d_model, n_head, d_ff, dropout)
                 for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, n_layers,
                 d_ff, dropout, max_seq_len):
        super(Decoder, self).__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        self.layers = nn.ModuleList(
                [DecoderLayer(d_model, n_head, d_ff, dropout)
                 for _ in range(n_layers)])
        self.fc = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        x = self.fc(x)
        return x


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model,
                 n_head, n_layers, d_ff, dropout, max_seq_len):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, d_model, n_head,
                               n_layers, d_ff, dropout, max_seq_len)
        self.decoder = Decoder(tgt_vocab_size, d_model, n_head,
                               n_layers, d_ff, dropout, max_seq_len)

    def generate_mask(self, x, pad_idx):
        mask = (x != pad_idx).unsqueeze(1).unsqueeze(2)
        return mask

    def generate_subsequent_mask(self, x):
        sz = x.size(1)
        mask = torch.triu(torch.ones(sz, sz, device=x.device),
                          diagonal=1).bool()
        return mask

    def forward(self, src, tgt, src_pad_idx, tgt_pad_idx):
        src_mask = self.generate_mask(src, src_pad_idx)
        tgt_mask = self.generate_mask(tgt, tgt_pad_idx)
        tgt_mask = tgt_mask & (~self.generate_subsequent_mask(tgt))

        enc_output = self.encoder(src, src_mask)
        output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        return output


def train_step(model, optimizer, criterion, src, tgt,
               src_pad_idx, tgt_pad_idx):
    optimizer.zero_grad()
    output = model(src, tgt[:, :-1], src_pad_idx, tgt_pad_idx)
    loss = criterion(output.view(-1, VOCAB_SIZE), tgt[:, 1:].reshape(-1))
    loss.backward()
    optimizer.step()
    return loss.item()


def main():
    src_data = torch.randint(0, VOCAB_SIZE, (64, MAX_SEQ_LEN))
    tgt_data = torch.randint(0, VOCAB_SIZE, (64, MAX_SEQ_LEN))
    src_pad_idx = 0
    tgt_pad_idx = 0

    model = Transformer(VOCAB_SIZE, VOCAB_SIZE, D_MODEL, N_HEAD,
                        N_LAYERS, D_FF, DROPOUT, MAX_SEQ_LEN)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    for epoch in range(num_epochs):
        loss = train_step(model, optimizer, criterion, src_data,
                          tgt_data, src_pad_idx, tgt_pad_idx)
        print(f"Epoch: {epoch+1}/{num_epochs}, Loss: {loss:.4f}")


if __name__ == "__main__":
    main()
