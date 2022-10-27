## Standard libraries
import math

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product(q, k, v):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, which_linear):
        super().__init__()
        assert (
            embed_dim % num_heads == 0
        ), "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.which_linear = which_linear

        # Stack all weight matrices 1...h together for efficiency
        self.qkv_proj = self.which_linear(input_dim, 3 * embed_dim)
        self.o_proj = self.which_linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, return_attention=False):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(
            batch_size, seq_length, self.num_heads, 3 * self.head_dim
        )
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v)
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        return o


class EncoderBlock(nn.Module):
    def __init__(
        self, input_dim, num_heads, dim_feedforward, dropout, which_linear
    ):
        """
        Inputs:
            input_dim - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            dim_feedforward - Dimensionality of the hidden layer in the MLP
            dropout - Dropout probability to use in the dropout layers
        """
        super().__init__()

        self.which_linear = which_linear
        # Attention layer
        self.self_attn = MultiheadAttention(
            input_dim, input_dim, num_heads, which_linear
        )

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            self.which_linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            self.which_linear(dim_feedforward, input_dim),
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Attention part
        x_pre1 = self.norm1(x)
        attn_out = self.self_attn(x_pre1)
        x = x + self.dropout(attn_out)

        # MLP part
        x_pre2 = self.norm2(x)
        linear_out = self.linear_net(x_pre2)
        x = x + self.dropout(linear_out)

        return x


class RelationalReasoning(nn.Module):
    def __init__(self, num_layers, hidden_dim, **block_args):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderBlock(**block_args) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        for l in self.layers:
            x = l(x)

        x = self.norm(x)
        return x

    def get_attention_maps(self, x):
        attention_maps = []
        for l in self.layers:
            _, attn_map = l.self_attn(x, return_attention=True)
            attention_maps.append(attn_map)
            x = l(x)
        return attention_maps
