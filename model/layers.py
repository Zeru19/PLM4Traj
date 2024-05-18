import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class ContinuousEncoding(nn.Module):
    """
    A type of trigonometric encoding for encode continuous values into distance-sensitive vectors.
    """

    def __init__(self, embed_size):
        super().__init__()
        self.omega = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, embed_size))).float(),
                                  requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(embed_size).float(), requires_grad=True)
        self.div_term = math.sqrt(1. / embed_size)

    def forward(self, x):
        """
        :param x: input sequence for encoding, (batch_size, seq_len)
        :return: encoded sequence, shape (batch_size, seq_len, embed_size)
        """
        encode = x.unsqueeze(-1) * self.omega.reshape(1, 1, -1) + self.bias.reshape(1, 1, -1)
        encode = torch.cos(encode)
        return self.div_term * encode
    

class PositionalEncoding(nn.Module):
    """
    A type of trigonometric encoding for indicating items' positions in sequences.
    """

    def __init__(self, embed_size, max_len):
        super().__init__()

        pe = torch.zeros(max_len, embed_size).float()
        pe.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, embed_size, 2).float() * -(math.log(10000.0) / embed_size)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, position_ids=None):
        """
        Args:
            x: (B, T, d_model)
            position_ids: (B, T) or None

        Returns:
            (1, T, d_model) / (B, T, d_model)
        """
        if position_ids is None:
            return self.pe[:, :x.size(1)]
        else:
            batch_size, seq_len = position_ids.shape
            pe = self.pe[:, :seq_len, :]  # (1, T, d_model)
            pe = pe.expand((position_ids.shape[0], -1, -1))  # (B, T, d_model)
            pe = pe.reshape(-1, self.d_model)  # (B * T, d_model)
            position_ids = position_ids.reshape(-1, 1).squeeze(1)  # (B * T,)
            output_pe = pe[position_ids].reshape(batch_size, seq_len, self.d_model).detach()
            return output_pe
        

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal-based function used for encoding timestamps.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class TimeEmbed(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, time):
        return self.time_mlp(time)


class TrajEmbedding(nn.Module):
    def __init__(self, d_model, dis_feats=[], num_embeds=[], con_feats=[],
                 pre_embed=None, pre_embed_update=False, second_col=None):
        super().__init__()

        self.d_model = d_model
        self.dis_feats = dis_feats
        self.con_feats = con_feats
        self.second_col = second_col

        if len(dis_feats):
            assert len(dis_feats) == len(num_embeds), \
                'length of num_embeds list should be equal to the number of discrete features.'
            self.dis_embeds = nn.ModuleList([nn.Embedding(num_embed, d_model) for num_embed in num_embeds])
        else:
            self.dis_embeds = None

        if len(con_feats):
            # continuous encoding
            # self.con_embeds = nn.ModuleList([ContinuousEncoding(d_model) for _ in con_feats])
            # linear
            self.con_embeds = nn.Linear(len(con_feats), d_model)
        else:
            self.con_embeds = None

        if pre_embed is not None:
            self.dis_embeds[0].weight = nn.Parameter(torch.from_numpy(pre_embed),
                                                     requires_grad=pre_embed_update)

        if second_col is not None:
            self.time_embed = ContinuousEncoding(d_model)

    def forward(self, x):
        B, L, E_in = x.shape

        h = torch.zeros(B, L, self.d_model).to(x.device)
        if self.dis_embeds is not None:
            for dis_embed, dis_feat in zip(self.dis_embeds, self.dis_feats):
                h += dis_embed(x[..., dis_feat].long())
        # continuous encoding
        # if self.con_embeds is not None:
        #     for con_embed, con_feat in zip(self.con_embeds, self.con_feats):
        #         h += con_embed(x[..., con_feat].float())
        if self.con_embeds is not None:
            h += self.con_embeds(x[..., self.con_feats].float())

        if self.second_col is not None:
            h += self.time_embed(x[..., int(self.second_col)])

        return h


class TrajConvEmbedding(nn.Module):
    def __init__(self, d_model, dis_feats=[], num_embeds=[], con_feats=[], kernel_size=3,
                 pre_embed=None, pre_embed_update=False, second_col=None):
        super().__init__()

        self.d_model = d_model
        self.dis_feats = dis_feats
        self.con_feats = con_feats
        self.second_col = second_col

        # Operates discrete features by look-up table.
        if len(dis_feats):
            assert len(dis_feats) == len(num_embeds), \
                'length of num_embeds list should be equal to the number of discrete features.'
            self.dis_embeds = nn.ModuleList([nn.Embedding(num_embed, d_model) for num_embed in num_embeds])
        else:
            self.dis_embeds = None

        if pre_embed is not None:
            self.dis_embeds[0].weight = nn.Parameter(torch.from_numpy(pre_embed),
                                                     requires_grad=pre_embed_update)

        # Operates continuous features by convolution.
        self.conv = nn.Conv1d(len(con_feats), d_model,
                              kernel_size=kernel_size, padding=(kernel_size - 1)//2)

        # Time embedding
        if second_col is not None:
            self.time_embed = ContinuousEncoding(d_model)

    def forward(self, x):
        B, L, E_in = x.shape

        h = torch.zeros(B, L, self.d_model).to(x.device)
        if self.dis_embeds is not None:
            for dis_embed, dis_feat in zip(self.dis_embeds, self.dis_feats):
                h += dis_embed(x[..., dis_feat].long())
        if self.con_feats is not None:
            h += self.conv(x[..., self.con_feats].transpose(1, 2)).transpose(1, 2)

        if self.second_col is not None:
            h += self.time_embed(x[..., int(self.second_col)])

        return h
    

class MultiHeadCrossAttentionLayer(nn.Module):
    batch = 0

    def __init__(self, input_size, d_model, meaningful_anchors, virtual_anchors, n_heads, save_attn_map=False):
        super(MultiHeadCrossAttentionLayer, self).__init__()

        self.n_heads = n_heads
        self.d_model = d_model
        if meaningful_anchors is None:
            meaningful_anchors = nn.Parameter(torch.zeros(0, input_size), requires_grad=False)
        if virtual_anchors is None:
            virtual_anchors = nn.Parameter(torch.zeros(0, input_size), requires_grad=True)
        self.meaningful_anchors = meaningful_anchors
        self.virtual_anchors = virtual_anchors
        self.num_anchors = meaningful_anchors.size(0) + virtual_anchors.size(0)

        self.save_attn_map = save_attn_map

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.query = nn.Linear(input_size, d_model)
        self.key = nn.Linear(input_size, d_model)
        self.value = nn.Linear(input_size, d_model)

        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, L, E = x.shape
        anchors = torch.concat([self.meaningful_anchors, self.virtual_anchors], dim=0)  # (N, E)

        q = self.query(x).view(B, L, self.n_heads, self.d_model // self.n_heads).permute(0, 2, 1, 3)  # (B, n_heads, L, d_model/n_heads)
        k = self.key(anchors).view(self.num_anchors, self.n_heads, self.d_model // self.n_heads).permute(1, 0, 2)  # (n_heads, N, d_model/n_heads)
        v = self.value(anchors).view(self.num_anchors, self.n_heads, self.d_model // self.n_heads).permute(1, 0, 2)  # (n_heads, N, d_model/n_heads)

        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_model // self.n_heads, dtype=torch.float32))
        attn = F.softmax(scores, dim=-1)  # (B, n_heads, L, N)

        if self.save_attn_map:
            np.save(f'epoch0/attn_scores_{int(MultiHeadCrossAttentionLayer.batch)}', attn.detach().cpu().numpy())
            MultiHeadCrossAttentionLayer.batch += 1

        output = torch.matmul(attn, v)  # (B, n_heads, L, d_model/n_heads)
        output = output.permute(0, 2, 1, 3).contiguous().view(B, L, self.d_model)
        output = self.out_linear(output)

        return output
    

class PatternSemanticProjector(nn.Module):
    """ Project movement patterns onto a semantic-rich textual space. """
    
    def __init__(self, emb_size, d_model, meaningful_anchors, virtual_anchors, n_heads,
                 dropout=0.1, save_attn_map=False) -> None:
        super().__init__()

        self.mhca = MultiHeadCrossAttentionLayer(emb_size, d_model, meaningful_anchors, virtual_anchors, n_heads,
                                                 save_attn_map=save_attn_map)
        # feedforward layer
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        # self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        return x + self.ffn(self.mhca(x))
