#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import argparse
import os
import string
import copy

from collections import Counter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

EOS_TOKEN = "<eos>"
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"


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
        enc_attn_output, _ = self.enc_attn(x,
                                           enc_output,
                                           enc_output,
                                           src_mask)
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
    def __init__(self,
                 src_vocab_size,
                 tgt_vocab_size, d_model,
                 n_head,
                 n_layers,
                 d_ff,
                 dropout,
                 max_seq_len,
                 pad_idx,
                 eos_idx,
                 sos_idx):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, d_model, n_head,
                               n_layers, d_ff, dropout, max_seq_len)
        self.decoder = Decoder(tgt_vocab_size, d_model, n_head,
                               n_layers, d_ff, dropout, max_seq_len)
        self.pad_idx = pad_idx
        self.eos_idx = eos_idx
        self.sos_idx = sos_idx

    def generate_mask(self, x, pad_idx):
        mask = (x != pad_idx).unsqueeze(1).unsqueeze(2)
        return mask

    def generate_subsequent_mask(self, x):
        sz = x.size(1)
        mask = torch.triu(torch.ones(sz, sz, device=x.device),
                          diagonal=1).bool()
        return mask

    def forward(self, src, tgt):
        src_mask = self.generate_mask(src, self.pad_idx)
        tgt_mask = self.generate_mask(tgt, self.pad_idx)
        tgt_mask = tgt_mask & (~self.generate_subsequent_mask(tgt))

        enc_output = self.encoder(src, src_mask)
        output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        return output

    def generate_text(self, src, vocab, max_gen_len, max_seq_len):
        model = self
        model.eval()
        with torch.no_grad():
            enc_output = model.encoder(src,
                                       model.generate_mask(src, self.pad_idx))

            # Initialize target sequence with <sos> token
            tgt = torch.full((src.size(0), 1),
                             self.sos_idx,
                             dtype=torch.long,
                             device=src.device)
            tgt_save = copy.deepcopy(tgt)

            for i in range(max_gen_len):
                output = model.decoder(
                    tgt,
                    enc_output,
                    model.generate_mask(src, self.pad_idx),
                    model.generate_mask(tgt, self.pad_idx),
                )
                next_token = output[:, -1, :].argmax(dim=-1)

                if next_token.item() == self.pad_idx:
                    print("A pad token found.")
                    break

                if next_token.item() == self.eos_idx:
                    print("An eos token is found.")
                    break

                if tgt.size(1) >= max_seq_len:
                    tgt = tgt[:, 1:]

                tgt = torch.cat([tgt, next_token.unsqueeze(1)], dim=1)
                tgt_save = torch.cat([tgt_save, next_token.unsqueeze(1)],
                                     dim=1)

        generated_ids = tgt_save.squeeze().tolist()
        generated_tokens = [vocab[id_] for id_ in generated_ids
                            if id_ < len(vocab) and id_ != 0 and
                            id_ != self.sos_idx]
        generated_text = ' '.join(generated_tokens)
        return generated_text


def train_step(model, optimizer, criterion, src, tgt, vocab_size):
    optimizer.zero_grad()
    output = model(src, tgt[:, :-1])
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
    vocab = [PAD_TOKEN, EOS_TOKEN, SOS_TOKEN] + \
            [token for token, count in token_counts.items()]
    token_to_id = {token: id_ for id_, token in enumerate(vocab)}
    return vocab, token_to_id


def tokens_to_ids(tokens, token_to_id):
    return [token_to_id.get(token, token_to_id[PAD_TOKEN]) for token in tokens]


def pad_sequence(seq, max_len, pad_idx):
    seq = seq[:max_len]
    return seq + [pad_idx] * (max_len - len(seq))


def prepare_data(text_file, max_seq_len):
    with open(text_file, 'r', encoding='utf-8') as file:
        text = file.read()
    tokens = tokenize_text(text)
    vocab, token_to_id = build_vocab(tokens)
    pad_idx = token_to_id[PAD_TOKEN]
    eos_idx = token_to_id[EOS_TOKEN]
    sos_idx = token_to_id[SOS_TOKEN]

    # Add <sos> and <eos> tokens
    tokens_with_sos_eos = [SOS_TOKEN] + tokens + [EOS_TOKEN]

    src_sequences = []
    tgt_sequences = []
    for i in range(0, len(tokens_with_sos_eos) - max_seq_len):
        src_seq = tokens_with_sos_eos[i:i + max_seq_len]
        tgt_seq = tokens_with_sos_eos[i + 1:i + max_seq_len + 1]

        src_ids = tokens_to_ids(src_seq, token_to_id)
        tgt_ids = tokens_to_ids(tgt_seq, token_to_id)

        src_ids = pad_sequence(src_ids, max_seq_len, pad_idx)
        tgt_ids = pad_sequence(tgt_ids, max_seq_len, pad_idx)

        src_sequences.append(src_ids)
        tgt_sequences.append(tgt_ids)

    src = torch.tensor(src_sequences, dtype=torch.long)
    tgt = torch.tensor(tgt_sequences, dtype=torch.long)

    return src, tgt, pad_idx, eos_idx, sos_idx, len(vocab), vocab


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
    parser.add_argument("--d_model", type=int, default=128,
                        help="Dimension of model.")
    parser.add_argument("--n_head", type=int, default=4,
                        help="Number of heads in attention.")
    parser.add_argument("--n_layers", type=int, default=2,
                        help="Number of layers in encoder/decoder.")
    parser.add_argument("--d_ff", type=int, default=128,
                        help="Dimension of feedforward network.")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate.")
    parser.add_argument("--max_seq_len", type=int, default=10,
                        help="Maximum sequence length.")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay.")
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="Number of training epochs.")
    parser.add_argument("--smoothing_window", type=int, default=10,
                        help="Smoothing window for loss.")
    parser.add_argument("--max_gen_len", type=int, default=100,
                        help="Maximum length of generated text.")
    parser.add_argument("--save_path", type=str,
                        default="transformer_model.pt",
                        help="Path for save model.")
    parser.add_argument("--load_path", type=str, default="",
                        help="Path for load model.")

    args = parser.parse_args()

    if not os.path.exists(args.text_file):
        print(f"Error: The file '{args.text_file}' does not exist.")
        return

    src, tgt, pad_idx, eos_idx, sos_idx, vocab_size, vocab = prepare_data(
            args.text_file,
            args.max_seq_len)

    model = Transformer(vocab_size,
                        vocab_size,
                        args.d_model,
                        args.n_head,
                        args.n_layers,
                        args.d_ff,
                        args.dropout,
                        args.max_seq_len,
                        pad_idx,
                        eos_idx,
                        sos_idx)
    model.apply(init_weights)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate,
                            weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    # Load model if specified
    if args.load_path:
        if os.path.exists(args.load_path):
            model.load_state_dict(torch.load(args.load_path))
            print(f"Model loaded from {args.load_path}")
        else:
            print(f"Warning: Could not find model at {args.load_path}. " +
                  "Training from scratch.")
    else:
        print("Training from scratch")

    losses = []
    for epoch in range(args.num_epochs):
        loss = train_step(model, optimizer, criterion, src, tgt, vocab_size)
        losses.append(loss)

        if len(losses) >= args.smoothing_window:
            smoothed_loss = np.mean(losses[-args.smoothing_window:])
        else:
            smoothed_loss = np.mean(losses)
        scheduler.step(smoothed_loss)
        print(f"Epoch: {epoch+1}/{args.num_epochs}, Loss: {loss:.4f}, " +
              f"Smoothed Loss: {smoothed_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), args.save_path)
    print(f"Model saved at {args.save_path}")

    generated_text = model.generate_text(src[:1],
                                         vocab,
                                         args.max_gen_len,
                                         args.max_seq_len)
    print(f"\nGenerated Text: \n{generated_text}")


if __name__ == "__main__":
    main()
