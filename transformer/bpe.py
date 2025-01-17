import re
from collections import Counter


class BPE:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.merges = {}
        self.vocab = set()
        self.vocab_to_id = {}
        self.id_to_vocab = {}
        self.begin_token = "<begin>"
        self.end_token = "<end>"
        self.unk_token = "<unk>"

    def get_stats(self, words):
        pairs = Counter()
        for word, freq in words.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def merge_vocab(self, pair, v_in):
        v_out = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        for word in v_in:
            w_out = p.sub(''.join(pair), word)
            v_out[w_out] = v_in[word]
        return v_out

    def learn_merges(self, words):
        vocab = {word: freq for word, freq in words.items()}
        vocab["<begin>"] = 1
        vocab["<end>"] = 1

        for i in range(self.vocab_size -
                       len(set(" ".join(vocab.keys()).split()))):
            pairs = self.get_stats(vocab)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            vocab = self.merge_vocab(best, vocab)
            self.merges[best] = ''.join(best)
        self.vocab = set(" ".join(vocab.keys()).split() +
                         list(self.merges.values()))

        if self.begin_token in self.vocab:
            self.vocab.remove(self.begin_token)
            self.vocab = [self.begin_token] + list(self.vocab)
        if self.end_token in self.vocab:
            self.vocab.remove(self.end_token)
            self.vocab = [self.begin_token, self.end_token] + list(self.vocab)

        if self.unk_token not in self.vocab:
            self.vocab.append(self.unk_token)

        for i, token in enumerate(self.vocab):
            self.vocab_to_id[token] = i
            self.id_to_vocab[i] = token

    def tokenize(self, text):
        tokens = [self.begin_token]
        for word in text.split():
            current_tokens = []
            while word:
                found = False
                for token in sorted(self.vocab, key=len, reverse=True):
                    if word.startswith(token):
                        current_tokens.append(token)
                        word = word[len(token):]
                        found = True
                        break
                if not found:
                    current_tokens.append(self.unk_token)
                    break
            tokens.extend(current_tokens)
        tokens.append(self.end_token)
        return tokens
