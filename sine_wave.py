#!/usr/bin/env python3

import pdb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import argparse
import os
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


class Embedding(nn.Module):
    def __init__(self, input_size, d_model):
        super(Embedding, self).__init__()
        self.linear = nn.Linear(input_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.linear(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model

    def forward(self, x):
        seq_len = x.size(1)
        pe = torch.zeros(seq_len, self.d_model, device=x.device)
        position = torch.arange(0, seq_len, dtype=torch.float,
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

        # Linear transformations
        q = self.W_q(query)  # (B, N, d_model)
        k = self.W_k(key)  # (B, N, d_model)
        v = self.W_v(value)  # (B, N, d_model)

        # Split into heads
        q = q.view(batch_size, -1, self.n_head,
                   self.d_k).transpose(1, 2)  # (B, H, N, d_k)
        k = k.view(batch_size, -1, self.n_head,
                   self.d_k).transpose(1, 2)  # (B, H, N, d_k)
        v = v.view(batch_size, -1, self.n_head,
                   self.d_k).transpose(1, 2)  # (B, H, N, d_k)

        attn_output, attn_probs = scaled_dot_product_attention(q, k, v, mask)

        # Concatenate heads and linear transformation
        attn_output = attn_output.transpose(1, 2).contiguous().view(
                batch_size, -1, self.d_model)
        output = self.W_o(attn_output)  # (B, N, d_model)
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


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head)
        self.ffn = FeedForwardNetwork(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask, tgt_mask):
        attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x


class Decoder(nn.Module):
    def __init__(self, output_size, d_model, n_head, n_layers,
                 d_ff, dropout):
        super(Decoder, self).__init__()
        self.embedding = Embedding(1, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList(
                [DecoderLayer(d_model, n_head, d_ff, dropout)
                 for _ in range(n_layers)])
        self.fc = nn.Linear(d_model, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask, tgt_mask):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, src_mask, tgt_mask)
        x = self.fc(x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self,
                 output_size,
                 d_model,
                 n_head,
                 n_layers,
                 d_ff,
                 dropout):
        super(TransformerDecoder, self).__init__()
        self.decoder = Decoder(output_size, d_model, n_head,
                               n_layers, d_ff, dropout)
        self.output_size = output_size
        self.n_head = n_head

    def generate_subsequent_mask(self, tgt):
        size = tgt.size(1)
        subsequent_mask = (1 - torch.triu(
            torch.ones((self.n_head, size, size),
                       device=tgt.device), diagonal=1)).bool()
        return subsequent_mask

    def generate_mask(self, x):
        mask = torch.ones(x.size(0),
                          self.n_head,
                          x.size(1),
                          x.size(1),
                          device=x.device).bool()
        return mask

    def forward(self, tgt):
        tgt_mask = self.generate_subsequent_mask(tgt)
        src_mask = self.generate_mask(tgt)
        output = self.decoder(tgt, src_mask, tgt_mask)
        return output

    def generate_data(self, start_sequence, max_gen_len, seq_len):
        self.eval()
        generated_sequence = start_sequence.unsqueeze(0)

        with torch.no_grad():
            for _ in range(max_gen_len + 1):
                tgt = generated_sequence[:, -seq_len:]
                output = self(tgt)
                next_val = output[:, -1:]
                generated_sequence = torch.cat(
                        (generated_sequence, next_val), dim=1)

        return generated_sequence[:, 1:]


class SineWaveDataset(Dataset):
    def __init__(self, seq_len):
        self.seq_len = seq_len
        self.src_sequences, self.tgt_sequences = self._prepare_data()

    def _generate_sine_wave(self):
        time = np.arange(self.seq_len * 10)
        wave = np.sin(2 * np.pi * time / 10)
        return np.stack([wave], axis=-1)

    def _prepare_data(self):
        src_sequences = []
        tgt_sequences = []
        wave = self._generate_sine_wave()
        self.wave = wave
        for i in range(0, wave.size - self.seq_len):
            src_seq = wave[i:i + self.seq_len]
            tgt_seq = wave[i + 1:i + self.seq_len + 1, 0:1]

            src_sequences.append(src_seq)
            tgt_sequences.append(tgt_seq)

        return np.array(src_sequences, dtype=np.float32), \
            np.array(tgt_sequences, dtype=np.float32)

    def __len__(self):
        return len(self.src_sequences)

    def __getitem__(self, idx):
        src = torch.tensor(self.src_sequences[idx], dtype=torch.float)
        tgt = torch.tensor(self.tgt_sequences[idx], dtype=torch.float)
        return src, tgt


def train_step(model, optimizer, criterion, src, tgt, device):
    optimizer.zero_grad()
    output = model(tgt.to(device))
    loss = criterion(output, tgt.to(device))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.item()


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
    parser.add_argument("--d_model", type=int, default=32,
                        help="Dimension of model.")
    parser.add_argument("--n_head", type=int, default=4,
                        help="Number of heads in attention.")
    parser.add_argument("--n_layers", type=int, default=2,
                        help="Number of layers in decoder.")
    parser.add_argument("--d_ff", type=int, default=128,
                        help="Dimension of feedforward network.")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate.")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay.")
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="Number of training epochs.")
    parser.add_argument("--smoothing_window", type=int, default=10,
                        help="Smoothing window for loss.")
    parser.add_argument("--save_path", type=str,
                        default="transformer_model.pt",
                        help="Path for save model.")
    parser.add_argument("--load_path", type=str, default="",
                        help="Path for load model.")
    parser.add_argument("--seq_len", type=int, default=10,
                        help="Length of the Sine wave.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = SineWaveDataset(args.seq_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = TransformerDecoder(
        1,
        args.d_model,
        args.n_head,
        args.n_layers,
        args.d_ff,
        args.dropout,
    ).to(device)
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
        epoch_loss = 0
        for batch_idx, (src, tgt) in enumerate(dataloader):
            loss = train_step(model, optimizer, criterion, src, tgt, device)
            epoch_loss += loss
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                losses.append(epoch_loss / (batch_idx + 1))

        if len(losses) >= args.smoothing_window:
            smoothed_loss = np.mean(losses[-args.smoothing_window:])
        else:
            smoothed_loss = np.mean(losses)
        scheduler.step(smoothed_loss)
        print(f"Epoch: {epoch+1}/{args.num_epochs}, " +
              f"Loss: {epoch_loss/(batch_idx+1):.4f}, " +
              f"Smoothed Loss: {smoothed_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), args.save_path)
    print(f"Model saved at {args.save_path}")

    # Generate data using the first source sequence
    first_src, _ = dataset[0]
    first_src = first_src.to(device)

    generated_data = model.generate_data(first_src,
                                         dataset.src_sequences.shape[0],
                                         args.seq_len)
    generated_data = generated_data.cpu().numpy().squeeze()

    # Get the first input sequence from the training data
    original_wave = dataset.wave.squeeze(-1)

    time = np.arange(len(original_wave))
    plt.plot(time, original_wave,
             label='Original Sine Wave (input)')
    plt.plot(time + 1, generated_data,
             label='Predicted Sine Wave')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Original and Predicted Sine Wave')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
