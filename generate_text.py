#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
from transformers import AutoTokenizer, AutoModel
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"


def load_text(file_path):
    """Loads text from a file with error handling."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
        return text
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)


def prepare_data(tokenizer, text, max_seq_length):
    """Prepares the data for the model."""
    tokens = tokenizer.tokenize(text)
    data = []
    for i in range(0, len(tokens) - max_seq_length, 1):
        input_tokens = tokens[i:i + max_seq_length]
        target_tokens = tokens[i + 1:i + max_seq_length + 1]

        input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)

        data.append((input_ids, target_ids))

    return data


class TextDataset(Dataset):
    """Dataset for the text data."""

    def __init__(self, data, pad_token_id, max_seq_length):
        self.data = data
        self.pad_token_id = pad_token_id
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_ids, target_ids = self.data[idx]
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        target_ids = torch.tensor(target_ids, dtype=torch.long)

        if len(input_ids) < self.max_seq_length:
            pad_len = self.max_seq_length - len(input_ids)
            input_ids = torch.cat(
                [input_ids, torch.full((pad_len,), self.pad_token_id)]
            )
            target_ids = torch.cat(
                [target_ids, torch.full((pad_len,), self.pad_token_id)]
            )
        return input_ids, target_ids


def create_dataloader(data, pad_token_id, batch_size, max_seq_length,
                      shuffle=True):
    """Creates a DataLoader for the dataset."""
    dataset = TextDataset(data, pad_token_id, max_seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


class PositionalEncoding(nn.Module):
    """Positional Encoding module."""

    def __init__(self, embed_dim, max_seq_length):
        super().__init__()
        encoding = torch.zeros(max_seq_length, embed_dim)
        position = torch.arange(0,
                                max_seq_length,
                                dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() *
            (-torch.log(torch.tensor(10000.0)) / embed_dim)
        )
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        encoding = encoding.unsqueeze(0)
        self.register_buffer("positional_encoding", encoding)

    def forward(self, x):
        """Apply positional encodings."""
        return x + self.positional_encoding[:, :x.size(1), :]


class TextTransformer(nn.Module):
    """Text Transformer model (Encoder-Decoder)."""

    def __init__(self, model_name, vocab_size, max_seq_length, n_head,
                 n_layers, hidden_dim):
        super().__init__()
        self.pretrained_model = AutoModel.from_pretrained(model_name)
        embed_dim = self.pretrained_model.config.hidden_size
        self.embedding = self.pretrained_model.embeddings.word_embeddings
        self.positional_encoding = PositionalEncoding(
            embed_dim, max_seq_length)

        encoder_layers = TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_head, dim_feedforward=hidden_dim)
        self.transformer_encoder = TransformerEncoder(
            encoder_layers, num_layers=n_layers)

        decoder_layers = TransformerDecoderLayer(
            d_model=embed_dim, nhead=n_head, dim_feedforward=hidden_dim)
        self.transformer_decoder = TransformerDecoder(
            decoder_layers, num_layers=n_layers)

        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, src, tgt):
        src = self.embedding(src) * torch.sqrt(
            torch.tensor(src.size(-1), dtype=torch.float))
        src = self.positional_encoding(src)
        tgt = self.embedding(tgt) * torch.sqrt(
            torch.tensor(tgt.size(-1), dtype=torch.float))
        tgt = self.positional_encoding(tgt)

        encoder_output = self.transformer_encoder(
                src.transpose(0, 1)).transpose(0, 1)
        decoder_output = self.transformer_decoder(
                tgt.transpose(0, 1),
                encoder_output.transpose(0, 1)).transpose(0, 1)
        output = self.fc(decoder_output)
        return output


def train_epoch(model, dataloader, criterion, optimizer,
                device, clip_norm=1.0):
    """Trains the model for one epoch."""
    model.train()
    epoch_loss = 0
    for src_batch, tgt_batch in dataloader:
        src_batch = src_batch.to(device)
        tgt_batch = tgt_batch.to(device)
        optimizer.zero_grad()

        output = model(src_batch[:, :-1], tgt_batch[:, :-1])
        loss = criterion(output.reshape(-1, output.size(-1)),
                         tgt_batch[:, 1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def train(model, train_dataloader, criterion, optimizer,
          scheduler, device, num_epochs, clip_norm):
    """Trains the model."""
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_dataloader, criterion,
                                 optimizer, device, clip_norm)

        if isinstance(scheduler,
                      torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(train_loss)
        else:
            scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch: {epoch+1}/{num_epochs}, " +
              f"Train Loss: {train_loss:.4f}, " +
              f"LR: {current_lr:.6f}")

    print("Training finished!")


def generate_text(model, tokenizer, device, max_seq_length=128, start_text=""):
    """Generates text from the model."""
    model.eval()
    tokens = tokenizer.tokenize(start_text)

    if tokenizer.bos_token is None:
        bos_token = BOS_TOKEN
    else:
        bos_token = tokenizer.bos_token

    input_ids = tokenizer.convert_tokens_to_ids(
        [bos_token] + tokens)
    input_ids = torch.tensor(input_ids,
                             dtype=torch.long).unsqueeze(0).to(device)

    output_ids = input_ids.clone()

    with torch.no_grad():
        for _ in range(max_seq_length):
            output = model(input_ids, output_ids)
            next_token_id = torch.argmax(output[:, -1, :], dim=-1)

            output_ids = torch.cat((output_ids,
                                    next_token_id.unsqueeze(0)), dim=1)

            if next_token_id.item() == tokenizer.eos_token_id:
                break
            if len(output_ids[0]) >= max_seq_length:
                break

        generated_ids = output_ids.squeeze().tolist()

    generated_text = tokenizer.decode(generated_ids,
                                      skip_special_tokens=True)
    return generated_text.strip()


def load_model(model, model_path, device):
    """Loads the model from the given path."""
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model


def main():
    parser = argparse.ArgumentParser(description="Train a Transformer model.")
    parser.add_argument("text_file", type=str, help="Path to the text file.")
    parser.add_argument("--start_text",
                        type=str, default="",
                        help="Text to start generation from.")
    parser.add_argument("--batch_size",
                        type=int, default=16, help="Batch size.")
    parser.add_argument("--hidden_dim",
                        type=int, default=2048, help="Hidden dimension.")
    parser.add_argument("--n_head",
                        type=int, default=8, help="Number of attention heads.")
    parser.add_argument("--n_layers",
                        type=int, default=6,
                        help="Number of transformer layers.")
    parser.add_argument("--learning_rate",
                        type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--num_epochs",
                        type=int, default=100,
                        help="Number of training epochs.")
    parser.add_argument("--max_seq_length",
                        type=int, default=32, help="Maximum sequence length.")
    parser.add_argument("--model_name",
                        type=str, default="bert-base-uncased",
                        help="Name of the pretrained tokenizer model.")
    parser.add_argument("--weight_decay", type=float,
                        default=0.01,
                        help="Weight decay.")
    parser.add_argument("--clip_norm", type=float,
                        default=1.0,
                        help="Gradient clipping norm.")
    parser.add_argument("--train_val_split", type=float, default=0.8,
                        help="Proportion of data to use for training")
    parser.add_argument("--load_model", type=str, default=None,
                        help="Path to load a pretrained model")

    args = parser.parse_args()

    file_path = args.text_file
    start_text = args.start_text
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    NUM_EPOCHS = args.num_epochs
    MAX_SEQ_LENGTH = args.max_seq_length
    MODEL_NAME = args.model_name
    WEIGHT_DECAY = args.weight_decay
    CLIP_NORM = args.clip_norm
    TRAIN_VAL_SPLIT = args.train_val_split
    LOAD_MODEL_PATH = args.load_model
    HIDDEN_DIM = args.hidden_dim
    N_HEAD = args.n_head
    N_LAYERS = args.n_layers

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': PAD_TOKEN})

    if tokenizer.unk_token is None:
        tokenizer.add_special_tokens({'unk_token': UNK_TOKEN})

    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({'eos_token': EOS_TOKEN})

    text = load_text(file_path)
    data = prepare_data(tokenizer, text, MAX_SEQ_LENGTH)

    # Split data into training and validation sets
    split_idx = int(len(data) * TRAIN_VAL_SPLIT)
    train_data = data[:split_idx]

    train_dataloader = create_dataloader(train_data,
                                         tokenizer.pad_token_id,
                                         BATCH_SIZE,
                                         MAX_SEQ_LENGTH,
                                         shuffle=True)

    vocab_size = len(tokenizer)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = TextTransformer(
        MODEL_NAME, vocab_size, MAX_SEQ_LENGTH, N_HEAD, N_LAYERS, HIDDEN_DIM
    ).to(device)

    if LOAD_MODEL_PATH:
        model = load_model(model, LOAD_MODEL_PATH, device)
        print(f"Model loaded from {LOAD_MODEL_PATH}")

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                            weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=5, factor=0.5)

    train(model, train_dataloader, criterion, optimizer,
          scheduler, device, NUM_EPOCHS, CLIP_NORM)

    generated_text = generate_text(
        model, tokenizer, device, MAX_SEQ_LENGTH, start_text
    )
    print(f"Generated Text based on '{start_text}': {generated_text}")


if __name__ == "__main__":
    main()
