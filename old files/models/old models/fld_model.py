import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FLDAttention(nn.Module):
    def __init__(
        self,
        input_dim,
        parameters,
        latent_dim=16,
        embed_dim=16,
        num_heads=2,
        shared_out=True,
    ):
        super(FLDAttention, self).__init__()
        assert embed_dim % num_heads == 0
        self.embed_time = embed_dim
        self.embed_time_k = embed_dim // num_heads
        self.h = num_heads
        self.nhidden = latent_dim
        if shared_out:
            self.out = nn.Sequential(nn.Linear(input_dim * num_heads, latent_dim))
        else:
            self.out = nn.Parameter(
                torch.randn(1, parameters, input_dim * num_heads, latent_dim)
            )
            self.out_bias = nn.Parameter(torch.zeros(1, parameters, latent_dim))
        self.shared = shared_out

        self.query_map = nn.Linear(embed_dim, embed_dim)
        self.key_map = nn.Linear(embed_dim, embed_dim)

    def attention(self, query, key, value, mask=None):
        dim = value.size(-1)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        scores = scores.unsqueeze(-1).repeat_interleave(dim, dim=-1)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(-3) == 0, float('-inf'))
        p_attn = F.softmax(scores, dim=-2)
        return torch.sum(p_attn * value.unsqueeze(-3), -2), p_attn

    def forward(self, query, key, value, mask=None):
        batch, seq_len, dim = value.size()
        if mask is not None:
            mask = mask.unsqueeze(1)
        value = value.unsqueeze(1)
        query = self.query_map(query)
        key = self.key_map(key)
        query, key = [
            x.view(x.size(0), -1, self.h, self.embed_time_k).transpose(1, 2)
            for x in (query, key)
        ]
        x, _ = self.attention(query, key, value, mask)
        x = x.transpose(1, 2).contiguous().view(batch, -1, self.h * dim)
        if self.shared:
            return self.out(x)
        else:
            x = x.unsqueeze(-2) @ self.out
            x = x.squeeze(-2) + self.out_bias
            return x


class FLD(nn.Module):
    def __init__(
        self,
        input_dim,
        latent_dim,
        embed_dim_per_head,
        num_heads,
        function,
        device,
        depth=1,
        hidden_dim=None,
        shared_out_for_attn=True
    ):
        super(FLD, self).__init__()
        if function == "C":
            P = 1
        elif function == "L":
            P = 2
        elif function == "Q":
            P = 3
        elif function == "S":
            P = 4
        else:
            raise ValueError(f"Unsupported function type: {function}")
        self.F = function
        embed_dim = embed_dim_per_head * num_heads
        self.attn = FLDAttention(
            input_dim=2 * input_dim,
            parameters=P,
            latent_dim=latent_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            shared_out=shared_out_for_attn,
        )
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.time_embedding = nn.Linear(1, embed_dim)
        self.query = nn.Parameter(torch.randn(1, P, embed_dim))
        if not hidden_dim:
            hidden_dim = latent_dim
        if depth > 0:
            layers = [nn.Linear(latent_dim, hidden_dim), nn.ReLU()]
            for _ in range(depth - 1):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
            layers.append(nn.Linear(hidden_dim, input_dim))
        else:
            layers = [nn.Linear(latent_dim, input_dim)]
        self.out = nn.Sequential(*layers)
        self.device = device
        self.latent_dim = latent_dim

    def learn_time_embedding(self, tt):
        tt = tt.to(self.device).unsqueeze(-1)
        inds = [i for i in range(self.embed_dim) if i % self.num_heads == 0]
        out = self.time_embedding(tt)
        out[:, :, inds] = torch.sin(out[:, :, inds])
        return out

    def forward(self, timesteps, X, M, y_time_steps):
        key = self.learn_time_embedding(timesteps).unsqueeze(1)
        X_cat = torch.cat((X, M), dim=-1)
        M_cat = torch.cat((M, M), dim=-1)
        coeffs = self.attn(self.query, key, X_cat, M_cat)
        if self.F == "C":
            x = coeffs[:, 0, :].unsqueeze(1).repeat(1, y_time_steps.size(1), 1)
        elif self.F == "Q":
            x = (
                coeffs[:, 0, :].unsqueeze(-2)
                + (y_time_steps.unsqueeze(-1) @ coeffs[:, 1, :].unsqueeze(-2))
                + ((y_time_steps.unsqueeze(-1) ** 2) @ coeffs[:, 2, :].unsqueeze(-2))
            )
        elif self.F == "L":
            x = coeffs[:, 0, :].unsqueeze(-2) + (y_time_steps.unsqueeze(-1) @ coeffs[:, 1, :].unsqueeze(-2))
        elif self.F == "S":
            x = (
                coeffs[:, 0, :].unsqueeze(-2)
                * torch.sin(coeffs[:, 1, :].unsqueeze(-2) * y_time_steps.unsqueeze(-1)
                            + coeffs[:, 2, :].unsqueeze(-2))
            ) + coeffs[:, 3, :].unsqueeze(-2)
        else:
            raise ValueError(f"Unsupported function type: {self.F}")
        return self.out(x)

__all__ = ["FLDAttention", "FLD"]
