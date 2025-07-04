import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class FLDAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,          # E: total embedding dimension (h * k)
        latent_dim: int,         # output dimension of each coefficient
        num_heads: int = 2,      # number of attention heads
        shared_out: bool = True, # share final projection
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.h = num_heads                  # h
        self.k = embed_dim // num_heads     # k = E/h
        self.embed_dim = embed_dim          # E

        # Linear maps for Q, K, V
        self.Wq = nn.Linear(embed_dim, embed_dim)  # maps Q→E
        self.Wk = nn.Linear(embed_dim, embed_dim)  # maps K→E
        self.Wv = nn.Linear(embed_dim, embed_dim)  # maps V→E

        if shared_out:
            # final projection from concatenated heads to latent_dim
            self.Wo = nn.Linear(embed_dim, latent_dim)
        else:
            # one Wo per basis (unused here)
            self.Wo = nn.ModuleList([
                nn.Linear(embed_dim, latent_dim) for _ in range(latent_dim)
            ])

    def forward(self, Q, K, V, mask=None):
        """
        Q: [B, P, E]    # P = number of bases
        K: [B, S, E]    # S = T*C sequence length
        V: [B, S, E]
        mask: [B, S]    # boolean mask over sequence

        returns coeffs: [B, P, latent_dim]
        """
        B, P, E = Q.shape
        _, S, _ = K.shape

        # -- project to multi-head space
        # Qp: [B, P, E] → [B, h, P, k]
        Qp = self.Wq(Q).view(B, P, self.h, self.k).permute(0,2,1,3)
        # Kp: [B, S, E] → [B, h, S, k]
        Kp = self.Wk(K).view(B, S, self.h, self.k).permute(0,2,1,3)
        # Vp: [B, S, E] → [B, h, S, k]
        Vp = self.Wv(V).view(B, S, self.h, self.k).permute(0,2,1,3)

        # -- scaled dot-product
        # scores: [B,h,P,S]
        scores = torch.einsum("bhpk,bhsk->bhps", Qp, Kp) / math.sqrt(self.k)
        if mask is not None:
            # mask: [B,1,1,S]
            m = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~m, float("-inf"))
        A = F.softmax(scores, dim=-1)  # [B,h,P,S]

        # -- context
        # C: [B,h,P,k]
        C = torch.einsum("bhps,bhsk->bhpk", A, Vp)
        # flatten: [B,P,E]
        C = C.permute(0,2,1,3).contiguous().view(B, P, E)

        # -- final projection
        if isinstance(self.Wo, nn.ModuleList):
            # per-basis projections
            out = torch.stack([self.Wo[i](C[:,i]) for i in range(P)], dim=1)
        else:
            # shared projection
            out = self.Wo(C)  # [B,P,latent_dim]
        return out

class FLD(nn.Module):
    def __init__(
        self,
        input_dim: int,         # C
        latent_dim: int,        # latent output dim
        num_heads: int,         # h
        embed_dim: int,         # E
        function: str,          # 'C','L','Q','S'
        residual_cycle: bool=False,
        cycle_length: int=24,
    ):
        super().__init__()
        self.residual_cycle = residual_cycle
        self.cycle_length = cycle_length

        # number of functional parameters
        if function == "C": P = 1
        elif function == "L": P = 2
        elif function == "Q": P = 3
        elif function == "S": P = 4
        self.P = P
        self.F = function
        self.E = embed_dim

        # time embedding: Linear(1→E)
        self.time_linear = nn.Linear(1, embed_dim)
        # channel embedding: [C, E]
        self.channel_embed = nn.Parameter(torch.randn(input_dim, embed_dim))

        # 3D attention
        self.attn = FLDAttention(embed_dim, latent_dim, num_heads)

        # decoder MLP: latent_dim→C
        self.out = nn.Sequential(nn.Linear(latent_dim, input_dim))

    def learn_time(self, tt):
        """tt: [B,T] → [B,T,E] with sin on half dims"""
        E_lin = self.time_linear(tt.unsqueeze(-1))  # [B,T,E]
        return E_lin + torch.sin(E_lin)

    def forward(self, timesteps, X, M, y_times):
        """
        timesteps: [B,T]
        X: [B,T,C]
        M: [B,T,C]
        y_times: [T_out]

        returns Ŷ: [B,T_out,C]
        """
        B,T,C = X.shape

        # -- residual cycle
        if self.residual_cycle:
            # c_in: [B,T,C], c_base: [B,cycle_length,C]
            c_in, c_base = self._get_cycle(timesteps, X, M)
            X = X - c_in

        # -- build [B,T,C,E]
        Et = self.learn_time(timesteps).unsqueeze(2).expand(-1,-1,C,-1)  # time keys
        Ec = self.channel_embed.unsqueeze(0).unsqueeze(0).expand(B,T,-1,-1)
        E_all = Et + Ec  # [B,T,C,E]

        # -- flatten to sequence
        S = T * C
        K = E_all.view(B, S, self.E)  # keys
        V = K.clone()                 # values
        mask = M.any(-1).view(B,T,1).expand(-1,-1,C).reshape(B,S)

        # -- queries: [B,P,E]
        Q = nn.Parameter(torch.randn(B, self.P, self.E), requires_grad=False).to(X.device)

        # -- attention
        coeffs = self.attn(Q, K, V, mask)  # [B,P,latent_dim]

        # -- reconstruct & decode (omitted)
        out = self.out(coeffs.mean(1)).unsqueeze(1).expand(-1,y_times.size(1),-1)
        return out