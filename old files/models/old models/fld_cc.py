
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class FLDAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        latent_dim: int,
        num_heads: int = 2,
        shared_out: bool = True,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.h = num_heads
        self.k = embed_dim // num_heads
        self.embed_dim = embed_dim

        # projections
        self.Wq = nn.Linear(embed_dim, embed_dim)
        self.Wk = nn.Linear(embed_dim, embed_dim)
        self.Wv = nn.Linear(embed_dim, embed_dim)

        if shared_out:
            self.Wo = nn.Linear(embed_dim, latent_dim)
        else:
            self.Wo = nn.ModuleList([nn.Linear(embed_dim, latent_dim) for _ in range(latent_dim)])

    def forward(self, Q, K, V, mask):
        # Q: [B, P, E], K/V: [B, S, E], mask: [B, S]
        B, P, E = Q.shape
        _, S, _ = K.shape

        Qp = self.Wq(Q).view(B, P, self.h, self.k).permute(0,2,1,3)   # [B,h,P,k]
        Kp = self.Wk(K).view(B, S, self.h, self.k).permute(0,2,1,3)   # [B,h,S,k]
        Vp = self.Wv(V).view(B, S, self.h, self.k).permute(0,2,1,3)   # [B,h,S,k]

        scores = torch.einsum("bhpk,bhsk->bhps", Qp, Kp) / math.sqrt(self.k)  # [B,h,P,S]
        mask = mask.unsqueeze(1).unsqueeze(2)  # [B,1,1,S]
        scores = scores.masked_fill(~mask, float("-inf"))
        A = F.softmax(scores, dim=-1)  # [B,h,P,S]

        C = torch.einsum("bhps,bhsk->bhpk", A, Vp)   # [B,h,P,k]
        C = C.permute(0,2,1,3).contiguous().view(B, P, E)  # [B,P,E]

        if isinstance(self.Wo, nn.ModuleList):
            out = torch.stack([self.Wo[i](C[:,i]) for i in range(P)], dim=1)  # [B,P,latent]
        else:
            out = self.Wo(C)  # [B,P,latent]
        return out

class FLD(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        num_heads: int,
        embed_dim: int,
        function: str,
        residual_cycle: bool = False,
        cycle_length: int = 24,
    ):
        super().__init__()
        self.residual_cycle = residual_cycle
        self.cycle_length = cycle_length

        if function == "C": P = 1
        elif function == "L": P = 2
        elif function == "Q": P = 3
        elif function == "S": P = 4
        else: raise ValueError(function)
        self.P = P
        self.F = function
        self.E = embed_dim

        self.time_linear = nn.Linear(1, embed_dim)
        self.channel_embed = nn.Parameter(torch.randn(input_dim, embed_dim))
        self.attn = FLDAttention(embed_dim, latent_dim, num_heads)

        layers = []
        layers.append(nn.Linear(latent_dim, input_dim))
        self.out = nn.Sequential(*layers)

    def _get_cycle(self, timesteps, X, M):
        B, T, C = X.shape
        t_floor = torch.floor(timesteps).long().to(X.device) % self.cycle_length
        c_base = torch.zeros(B, self.cycle_length, C, device=X.device)
        for p in range(self.cycle_length):
            mask_p = (t_floor == p).unsqueeze(-1) & M.bool()
            sum_p = (X * mask_p).sum(dim=1)
            cnt_p = mask_p.sum(dim=1).clamp(min=1)
            c_base[:, p] = sum_p / cnt_p
        c_in = c_base.gather(1, t_floor.unsqueeze(-1).expand(-1,-1,C))
        return c_in, c_base

    def learn_time(self, tt):
        E_lin = self.time_linear(tt.unsqueeze(-1))
        return E_lin + torch.sin(E_lin)

    def forward(self, timesteps, X, M, y_times):
        B, T, C = X.shape

        if self.residual_cycle:
            c_in, c_base = self._get_cycle(timesteps, X, M)
            X = X - c_in

        # embeddings [B,T,C,E]
        Et = self.learn_time(timesteps).unsqueeze(2).expand(-1,-1,C,-1)
        Ec = self.channel_embed.unsqueeze(0).unsqueeze(0).expand(B,T,-1,-1)
        E_all = Et + Ec

        # flatten to [B, S, E], S = T*C
        S = T * C
        K = E_all.view(B, S, self.E)
        V = K.clone()
        M_flat = M.any(-1).view(B, T,1).expand(-1,-1,C).view(B, S)

        # queries [B,P,E]
        Q = nn.Parameter(torch.randn(B, self.P, self.E), requires_grad=False).to(X.device)

        coeffs = self.attn(Q, K, V, M_flat)

        return self.out(coeffs.mean(1).unsqueeze(1).expand(-1,y_times.size(1),-1))
