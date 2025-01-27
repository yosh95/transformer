#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import argparse
import os
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy
import matplotlib.pyplot as plt


class Embedding(nn.Module):
    def __init__(self, input_size, d_model):
        super(Embedding, self).__init__()
        self.linear = nn.Linear(input_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.linear(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

    def forward(self, x):
        seq_len = x.size(1)
        pe = torch.zeros(self.max_seq_len, self.d_model, device=x.device)
        position = torch.arange(0, self.max_seq_len, dtype=torch.float,
                                device=x.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,
                                          self.d_model,
                                          2,
                                          device=x.device).float() *
                             (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return x + pe[:, :seq_len]


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
    def __init__(self, input_size, d_model, n_head,
                 n_layers, d_ff, dropout, max_seq_len):
        super(Encoder, self).__init__()
        self.embedding = Embedding(input_size, d_model)
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
    def __init__(self, output_size, d_model, n_head, n_layers,
                 d_ff, dropout, max_seq_len):
        super(Decoder, self).__init__()
        self.embedding = Embedding(1, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        self.layers = nn.ModuleList(
                [DecoderLayer(d_model, n_head, d_ff, dropout)
                 for _ in range(n_layers)])
        self.fc = nn.Linear(d_model, output_size)
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
                 input_size,
                 output_size,
                 d_model,
                 n_head,
                 n_layers,
                 d_ff,
                 dropout,
                 max_seq_len):
        super(Transformer, self).__init__()
        self.encoder = Encoder(input_size, d_model, n_head,
                               n_layers, d_ff, dropout, max_seq_len)
        self.decoder = Decoder(output_size, d_model, n_head,
                               n_layers, d_ff, dropout, max_seq_len)
        self.output_size = output_size
        self.max_seq_len = max_seq_len

    def generate_mask(self, x):
        mask = torch.ones(x.size(0), 1, 1, x.size(1), device=x.device).bool()
        return mask

    def generate_subsequent_mask(self, x):
        sz = x.size(1)
        mask = torch.triu(torch.ones(sz, sz, device=x.device),
                          diagonal=1).bool()
        return mask

    def forward(self, src, tgt):
        src_mask = self.generate_mask(src)
        tgt_mask = self.generate_mask(tgt)
        tgt_mask = tgt_mask & (~self.generate_subsequent_mask(tgt))

        enc_output = self.encoder(src, src_mask)
        output = self.decoder(tgt.unsqueeze(-1),
                              enc_output,
                              src_mask,
                              tgt_mask)
        return output.squeeze(-1)

    def generate_data(self, src, max_gen_len, max_seq_len, init_val):
        model = self
        model.eval()
        with torch.no_grad():
            enc_output = model.encoder(src, model.generate_mask(src))

            # Initialize target sequence with specified start value
            tgt = torch.full((src.size(0), 1),
                             init_val,
                             dtype=torch.float,
                             device=src.device)
            tgt_save = copy.deepcopy(tgt)
            for i in range(max_gen_len):
                output = model.decoder(tgt.unsqueeze(-1),
                                       enc_output,
                                       model.generate_mask(src),
                                       model.generate_mask(tgt))

                next_value = output[:, -1, :]

                if tgt.size(1) >= max_seq_len:
                    tgt = tgt[:, 1:]

                tgt = torch.cat([tgt, next_value], dim=1)
                tgt_save = torch.cat([tgt_save, next_value], dim=1)

        generated_values = tgt_save.squeeze().tolist()
        return generated_values


def train_step(model, optimizer, criterion, src, tgt):
    optimizer.zero_grad()
    output = model(src, tgt[:, :-1])
    loss = criterion(output, tgt[:, 1:])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.item()


def generate_sine_wave(length, period=10, amplitude=1, amplitude_modulator=1):
    time = np.arange(length)
    wave = amplitude * np.sin(2 * np.pi * time / period)
    amplitude_modulator_wave = amplitude_modulator * np.cos(
            2 * np.pi * time / period)
    return np.stack([wave, amplitude_modulator_wave], axis=-1)


def prepare_data(seq_len, num_data, max_seq_len):
    """
    Generates training data for sine wave prediction.

    Args:
        seq_len (int): The total length of each sine wave.
        num_data (int): The number of sine waves to generate.
        max_seq_len (int): The maximum sequence length for the Transformer.

    Returns:
        tuple: A tuple containing the source (src) and target (tgt) tensors.
    """
    src_sequences = []
    tgt_sequences = []
    for _ in range(num_data):
        # Generate sine wave data
        wave = generate_sine_wave(seq_len + max_seq_len)
        # Create input and target sequences using sliding window.
        for i in range(0, seq_len):
            src_seq = wave[i:i + max_seq_len]
            tgt_seq = wave[i + 1:i + max_seq_len + 1]

            src_sequences.append(src_seq)
            tgt_sequences.append(tgt_seq)

    # Convert lists of numpy arrays to PyTorch tensors
    src = torch.tensor(src_sequences, dtype=torch.float)
    tgt = torch.tensor(tgt_sequences, dtype=torch.float)

    return src, tgt


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        torch.nn.init.xavier_uniform_(m.weight)


def main():
    parser = argparse.ArgumentParser(
            description="Train a Transformer model to generate a sine wave.")
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
    parser.add_argument("--seq_len", type=int, default=100,
                        help="Length of the Sine wave.")
    parser.add_argument("--num_data", type=int, default=100,
                        help="Number of generated data")
    parser.add_argument("--init_val", type=float, default=0,
                        help="Start value for generation")

    args = parser.parse_args()

    src, tgt = prepare_data(args.seq_len, args.num_data, args.max_seq_len)

    model = Transformer(
        2,
        1,
        args.d_model,
        args.n_head,
        args.n_layers,
        args.d_ff,
        args.dropout,
        args.max_seq_len,
    )
    model.apply(init_weights)
    criterion = nn.MSELoss()
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
        loss = train_step(model, optimizer, criterion, src, tgt)
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

    generated_data = model.generate_data(src[:1, :, :],
                                         args.max_gen_len,
                                         args.max_seq_len,
                                         args.init_val)

    # Get the first input sequence from the training data
    original_wave = src[0].tolist()

    time = np.arange(len(original_wave) + len(generated_data) - 1)
    plt.plot(time[:len(original_wave)], [x[0] for x in original_wave],
             label='Original Sine Wave (input)')
    plt.plot(time[len(original_wave)-1:], generated_data,
             label='Predicted Sine Wave')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Original and Predicted Sine Wave')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
