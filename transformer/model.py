import torch.nn as nn

from transformer.positinal_encoding import PositionalEncoding


class Model(nn.Module):
    def __init__(self,
                 d_model,
                 num_layers,
                 input_size,
                 output_size,
                 nhead=4,
                 dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.input_linear = nn.Linear(input_size, d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model,
                                       nhead,
                                       d_model * 4,
                                       dropout,
                                       batch_first=True),
            num_layers
        )
        self.output_linear = nn.Linear(d_model, output_size)
        self.pos_encoder = PositionalEncoding(d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif p.dim() == 1:
                nn.init.zeros_(p)

    def forward(self, x):
        x = self.input_linear(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.output_linear(x)
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
