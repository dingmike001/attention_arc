import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
import math


class AttentiontTransformer(nn.Module):
    def __init__(self, eeg_channel: int, d_model: int, n_head: int, d_hid: int, n_layers: int, dropout: float = 0.5,
                 max_len: int = 799, device='cpu'):
        super(AttentiontTransformer, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=max_len)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=d_hid,
                                                    dropout=dropout, batch_first=True, norm_first=False, device=device)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)
        self.d_model = d_model
        self.lin1 = nn.Linear(d_model, 10)
        nn.init.xavier_normal_(self.lin1.weight)
        self.lin2 = nn.Linear(10, 3)
        nn.init.xavier_normal_(self.lin2.weight)
        self.sftmax = nn.Softmax(dim=1)
        self.conv2d = nn.Conv2d(eeg_channel, d_model, kernel_size=(3, 1), stride=(2, 1))
        self.dropf = nn.Dropout(p=dropout)
        self.liniar_in = nn.Linear(d_model, d_model)

    def decoder(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.sftmax(x)
        return x

    def forward(self, src):
        src = src.unsqueeze(-1)
        src = F.relu(self.conv2d(src))
        src = self.dropf(src)
        src = src.squeeze(-1)
        src = torch.transpose(src, 2, 1)
        src = self.liniar_in(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        output = output[:, -1, :]
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int = 32, dropout: float = 0.1, max_len: int = 1000):
        super().__init__()

        self.dropf = nn.Dropout(p=dropout)

        flag = (d_model % 2 == 0)
        if flag:
            d_model_new = d_model
        else:
            d_model_new = d_model + 1

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model_new, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model_new)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        if not flag:
            pe = pe[:, :, d_model - 1]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len,  embedding_dim]
        """
        a = self.pe[:, :x.size(1), :]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropf(x)


