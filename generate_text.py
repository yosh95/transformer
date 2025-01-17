#!/usr/bin/env python3

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import Counter
from torch.utils.data import Dataset, DataLoader
from transformer.bpe import BPE
from transformer.model import Model


def main():
    parser = argparse.ArgumentParser(
            description="Train and generate text with Transformer.")
    parser.add_argument("text_file", type=str, help="Path to the text file.")
    parser.add_argument("--start_text",
                        type=str,
                        default="",
                        help="Starting text for generation.")
    parser.add_argument("--vocab_size",
                        type=int,
                        default=3000,
                        help="Size of vocabulary.")
    parser.add_argument("--seq_len",
                        type=int,
                        default=64,
                        help="Sequence Length.")
    parser.add_argument("--batch_size",
                        type=int,
                        default=32,
                        help="Batch Size.")
    parser.add_argument("--d_model",
                        type=int,
                        default=32,
                        help="Model Dimension.")
    parser.add_argument("--num_layers",
                        type=int,
                        default=2,
                        help="Number of Layers.")
    parser.add_argument("--epochs",
                        type=int,
                        default=1000,
                        help="Number of epochs.")
    parser.add_argument("--lr",
                        type=float,
                        default=0.001,
                        help="Learning Rate.")
    parser.add_argument("--max_length",
                        type=int,
                        default=100,
                        help="Maximum Length for text generation.")
    parser.add_argument("--temperature",
                        type=float,
                        default=1.0,
                        help="Temperature for text generation.")
    parser.add_argument("--output_model",
                        type=str,
                        default="model.pth",
                        help="Path to save trained model.")
    parser.add_argument("--load_model",
                        type=str,
                        default=None,
                        help="Path to load a pre-trained model.")
    args = parser.parse_args()

    with open(args.text_file, "r") as f:
        text = f.read().strip()

    words = Counter(text.split())

    vocab_size = args.vocab_size
    bpe = BPE(vocab_size=vocab_size)
    bpe.learn_merges(words)

    tokens = bpe.tokenize(text)

    class TextDataset(Dataset):
        def __init__(self, tokens, vocab, seq_len):
            self.tokens = tokens
            self.vocab = list(vocab)
            self.vocab_to_id = {token: id for id,
                                token in enumerate(self.vocab)}
            self.ids = [self.vocab_to_id.get(token, 0)
                        for token in self.tokens]
            self.seq_len = seq_len

            self.sequences = []
            for i in range(0, len(self.ids) - seq_len, seq_len):
                self.sequences.append(self.ids[i: i + seq_len + 1])

        def __len__(self):
            return len(self.sequences)

        def __getitem__(self, idx):
            sequence = self.sequences[idx]
            return torch.tensor(
                    sequence[:-1],
                    dtype=torch.long), torch.tensor(sequence[1:],
                                                    dtype=torch.long)

    seq_len = args.seq_len
    dataset = TextDataset(tokens, bpe.vocab, seq_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    d_model = args.d_model
    num_layers = args.num_layers
    input_size = len(bpe.vocab)
    output_size = len(bpe.vocab)

    model = Model(d_model, num_layers, input_size, output_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device.type}")
    model.to(device)

    if args.load_model:
        try:
            model.load_state_dict(torch.load(args.load_model,
                                             map_location=device))
            print(f"Loaded model from {args.load_model}")
        except FileNotFoundError:
            print(f"Error: Model file not found at {args.load_model}")
            return
    else:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        epochs = args.epochs
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for inputs, targets in dataloader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                outputs = model(
                    nn.functional.one_hot(inputs,
                                          num_classes=len(bpe.vocab)).float())

                loss = criterion(outputs.transpose(1, 2), targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"Epoch: {epoch + 1}/{epochs}, " +
                  f"Loss: {total_loss / len(dataloader):.4f}")

        torch.save(model.state_dict(), args.output_model)
        print(f"Saved model to {args.output_model}")

    def generate_text(model,
                      start_text,
                      bpe,
                      device,
                      max_length=100,
                      temperature=1.0):
        model.eval()
        start_tokens = bpe.tokenize(start_text)
        start_ids = [bpe.vocab_to_id[token] for token in start_tokens]
        generated_ids = start_ids[:]

        with torch.no_grad():
            for _ in range(max_length):
                input_tensor = torch.tensor(
                        generated_ids).unsqueeze(0).to(device)
                output = model(
                    nn.functional.one_hot(
                            input_tensor,
                            num_classes=len(bpe.vocab)).float())
                output = output[:, -1, :] / temperature
                probs = F.softmax(output, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1).item()
                generated_ids.append(next_id)
                if next_id == bpe.vocab_to_id["<end>"]:
                    break

        generated_tokens = [bpe.id_to_vocab[idx] for idx in generated_ids]
        generated_text = " ".join(generated_tokens)
        return generated_text

    start_text = args.start_text
    generated_text = generate_text(model,
                                   start_text,
                                   bpe,
                                   device,
                                   args.max_length,
                                   args.temperature)
    print(f"\nGenerated Text: {generated_text}")


if __name__ == "__main__":
    main()
