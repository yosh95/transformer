#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import Transformer
import argparse

from transformers import AutoTokenizer

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"


def load_text(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
    return text


def prepare_data(tokenizer, text, max_seq_length):
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
                            [input_ids, torch.full((pad_len,),
                                                   self.pad_token_id)])
            target_ids = torch.cat(
                            [target_ids, torch.full((pad_len,),
                                                    self.pad_token_id)])
        return input_ids, target_ids


def create_dataloader(data, pad_token_id, batch_size, max_seq_length):
    dataset = TextDataset(data, pad_token_id, max_seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


class PositionalEncoding(nn.Module):

    def __init__(self, embed_dim, max_seq_length):
        super().__init__()
        self.encoding = torch.zeros(max_seq_length, embed_dim)
        position = torch.arange(0,
                                max_seq_length,
                                dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() *
            (-torch.log(torch.tensor(10000.0)) / embed_dim)
        )
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)
        self.register_buffer("positional_encoding", self.encoding)

    def forward(self, x):
        return x + self.positional_encoding[:, : x.size(1), :]


class TextTransformer(nn.Module):

    def __init__(self,
                 vocab_size,
                 embed_dim,
                 hidden_dim,
                 n_head,
                 n_layers,
                 max_seq_length):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(
                        embed_dim, max_seq_length)
        self.transformer = Transformer(
            embed_dim, n_head, n_layers, hidden_dim, batch_first=True
        )
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, src, tgt):
        src = self.embedding(src) * torch.sqrt(
                        torch.tensor(src.size(-1), dtype=torch.float))
        src = self.positional_encoding(src)
        tgt = self.embedding(tgt) * torch.sqrt(
                        torch.tensor(tgt.size(-1), dtype=torch.float))
        tgt = self.positional_encoding(tgt)
        output = self.transformer(src, tgt)
        output = self.fc(output)
        return output


def train(model, dataloader, criterion, optimizer, device, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for src_batch, tgt_batch in dataloader:
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)
            optimizer.zero_grad()

            output = model(src_batch[:, :-1], tgt_batch[:, :-1])
            loss = criterion(output.reshape(-1, output.size(-1)),
                             tgt_batch[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(dataloader)
        print(f"Epoch: {epoch+1}, Loss: {epoch_loss:.4f}")
    print("Training finished!")


def generate_text(model, tokenizer, device, max_seq_length=128, start_text=""):
    model.eval()
    with torch.no_grad():
        tokens = tokenizer.tokenize(start_text)

        if tokenizer.bos_token is None:
            bos_token = BOS_TOKEN
        else:
            bos_token = tokenizer.bos_token

        input_ids = tokenizer.convert_tokens_to_ids(
                        [bos_token] + tokens)
        input_ids = torch.tensor(input_ids,
                                 dtype=torch.long).unsqueeze(0).to(device)

        for _ in range(max_seq_length):
            output = model(input_ids[:, :-1], input_ids)
            next_token_id = torch.argmax(output[:, -1, :], dim=-1)

            input_ids = torch.cat((input_ids,
                                   next_token_id.unsqueeze(0)), dim=1)
            if next_token_id.item() == tokenizer.eos_token_id:
                break
        generated_ids = input_ids.squeeze().tolist()

        generated_text = tokenizer.decode(generated_ids,
                                          skip_special_tokens=True)
    return generated_text.strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Transformer model.")
    parser.add_argument("text_file", type=str, help="Path to the text file.")
    parser.add_argument("--start_text",
                        type=str, default="",
                        help="Text to start generation from.")
    parser.add_argument("--batch_size",
                        type=int, default=32, help="Batch size.")
    parser.add_argument("--embed_dim",
                        type=int, default=256, help="Embedding dimension.")
    parser.add_argument("--hidden_dim",
                        type=int, default=512, help="Hidden dimension.")
    parser.add_argument("--n_head",
                        type=int, default=4, help="Number of attention heads.")
    parser.add_argument("--n_layers",
                        type=int, default=2,
                        help="Number of transformer layers.")
    parser.add_argument("--learning_rate",
                        type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--num_epochs",
                        type=int, default=50,
                        help="Number of training epochs.")
    parser.add_argument("--max_seq_length",
                        type=int, default=128, help="Maximum sequence length.")
    parser.add_argument("--model_name",
                        type=str, default="bert-base-uncased",
                        help="Name of the pretrained tokenizer model.")

    args = parser.parse_args()

    file_path = args.text_file
    start_text = args.start_text
    BATCH_SIZE = args.batch_size
    EMBED_DIM = args.embed_dim
    HIDDEN_DIM = args.hidden_dim
    N_HEAD = args.n_head
    N_LAYERS = args.n_layers
    LEARNING_RATE = args.learning_rate
    NUM_EPOCHS = args.num_epochs
    MAX_SEQ_LENGTH = args.max_seq_length
    MODEL_NAME = args.model_name

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': PAD_TOKEN})

    if tokenizer.unk_token is None:
        tokenizer.add_special_tokens({'unk_token': UNK_TOKEN})

    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({'eos_token': EOS_TOKEN})

    # if tokenizer.bos_token is None:
    #     tokenizer.add_special_tokens({'bos_token': BOS_TOKEN})

    text = load_text(file_path)
    data = prepare_data(tokenizer, text, MAX_SEQ_LENGTH)
    dataloader = create_dataloader(data,
                                   tokenizer.pad_token_id,
                                   BATCH_SIZE,
                                   MAX_SEQ_LENGTH)
    vocab_size = len(tokenizer)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = TextTransformer(
        vocab_size, EMBED_DIM, HIDDEN_DIM, N_HEAD, N_LAYERS, MAX_SEQ_LENGTH
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train(model, dataloader, criterion, optimizer, device, NUM_EPOCHS)

    generated_text = generate_text(
        model, tokenizer, device, MAX_SEQ_LENGTH, start_text
    )
    print(f"Generated Text based on '{start_text}': {generated_text}")
