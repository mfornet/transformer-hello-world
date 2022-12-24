import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        nn.Embedding
        self.register_buffer("pe", pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(
        self,
        alpha,
        *,
        d_model=32,
        n_heads=8,
        n_layers=2,
        dropout=0.0,
        d_ff=32,
    ):
        super(Transformer, self).__init__()

        self.embedding = nn.Embedding(alpha, d_model)

        self.pe = PositionalEncoding(
            d_model=d_model, dropout=dropout, max_len=5000
        )

        decode_layer = nn.TransformerDecoderLayer(
            d_model, n_heads, dim_feedforward=d_ff, dropout=dropout
        )
        self.transformer = nn.TransformerDecoder(decode_layer, n_layers)
        self.linear = nn.Linear(d_model, alpha)

    def forward(self, src: torch.Tensor):
        src = self.embedding(src)
        src = self.pe(src)
        # TODO: pass src_mask to make sure that the model does not "cheat", and it can only see the previous tokens.
        output = self.transformer(src, src)
        output = self.linear(output)
        return F.log_softmax(output, dim=-1)