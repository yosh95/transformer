#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import argparse
import os
import string

from collections import Counter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

D_MODEL = 16
N_HEAD = 4
N_LAYERS = 2
D_FF = 16
DROPOUT = 0.1
MAX_SEQ_LEN = 100
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 2000
SMOOTHING_WINDOW = 10


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
               src_pad_idx, tgt_pad_idx, vocab_size):
    optimizer.zero_grad()
    output = model(src, tgt[:, :-1], src_pad_idx, tgt_pad_idx)
    loss = criterion(output.view(-1, vocab_size), tgt[:, 1:].reshape(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.item()


def tokenize_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.split()


def build_vocab(tokens):
    token_counts = Counter(tokens)
    vocab = ["<pad>", "<unk>"] + [token for token,
                                  count in token_counts.items()]
    token_to_id = {token: id_ for id_, token in enumerate(vocab)}
    return vocab, token_to_id


def tokens_to_ids(tokens, token_to_id):
    return [token_to_id.get(token, 1) for token in tokens]


def pad_sequence(seq, max_len, pad_idx):
    seq = seq[:max_len]
    return seq + [pad_idx] * (max_len - len(seq))


def prepare_data(text_file, max_seq_len):
    with open(text_file, 'r', encoding='utf-8') as file:
        text = file.read()
    tokens = tokenize_text(text)
    vocab, token_to_id = build_vocab(tokens)
    ids = tokens_to_ids(tokens, token_to_id)
    pad_idx = 0
    ids = pad_sequence(ids, max_seq_len, pad_idx)

    src = torch.tensor([ids], dtype=torch.long)
    tgt = torch.tensor([ids], dtype=torch.long)
    return src, tgt, pad_idx, len(vocab)


def generate_text(model, src, src_pad_idx, vocab, max_seq_len):
    model.eval()
    with torch.no_grad():
        tgt = torch.full((1, 1), src_pad_idx, dtype=torch.long)
        for i in range(max_seq_len - 1):
            output = model(src, tgt, src_pad_idx, src_pad_idx)
            next_token = output[:, -1, :].argmax(dim=-1)
            tgt = torch.cat([tgt, next_token.unsqueeze(1)], dim=1)

    generated_ids = tgt.squeeze().tolist()
    generated_tokens = [vocab[id_]
                        for id_ in generated_ids
                        if id_ < len(vocab) and id_ != 0]
    generated_text = ' '.join(generated_tokens)
    return generated_text


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        torch.nn.init.xavier_uniform_(m.weight)


def main():
    parser = argparse.ArgumentParser(
        description="Train a Transformer model to " +
                    "generate text from a text file.")
    parser.add_argument("text_file", type=str,
                        help="Path to the input text file.")
    args = parser.parse_args()

    if not os.path.exists(args.text_file):
        print(f"Error: The file '{args.text_file}' does not exist.")
        return

    src, tgt, pad_idx, vocab_size = prepare_data(args.text_file, MAX_SEQ_LEN)

    model = Transformer(vocab_size, vocab_size, D_MODEL, N_HEAD,
                        N_LAYERS, D_FF, DROPOUT, MAX_SEQ_LEN)
    model.apply(init_weights)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                            weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    losses = []
    for epoch in range(NUM_EPOCHS):
        loss = train_step(model, optimizer, criterion, src, tgt,
                          pad_idx, pad_idx, vocab_size)
        losses.append(loss)

        if len(losses) >= SMOOTHING_WINDOW:
            smoothed_loss = np.mean(losses[-SMOOTHING_WINDOW:])
        else:
            smoothed_loss = np.mean(losses)
        scheduler.step(smoothed_loss)
        print(f"Epoch: {epoch+1}/{NUM_EPOCHS}, Loss: {loss:.4f}, " +
              f"Smoothed Loss: {smoothed_loss:.4f}")

    vocab = build_vocab(tokenize_text(open(args.text_file,
                                           'r', encoding='utf-8').read()))[0]
    generated_text = generate_text(model, src, pad_idx, vocab, MAX_SEQ_LEN)
    print(f"\nGenerated Text: \n{generated_text}")


if __name__ == "__main__":
    main()
