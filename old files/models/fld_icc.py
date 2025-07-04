import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class FLDAttention(nn.Module):
    """
    Multi-head attention module used in Functional Latent Dynamics (FLD),
    operating over a flattened time-channel sequence.

    Args:
        embed_dim (int): Total embedding dimension (E).
        latent_dim (int): Output size of the attention module.
        num_heads (int): Number of attention heads.
        shared_out (bool): If True, use one projection for output.
                           If False, use separate projections per basis.
    """
    def __init__(self, embed_dim: int, latent_dim: int, num_heads: int = 2, shared_out: bool = True):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.h = num_heads                  # number of heads
        self.k = embed_dim // num_heads     # per-head dimension
        self.embed_dim = embed_dim          # total embedding dimension

        # Linear layers for query, key, value
        self.Wq = nn.Linear(embed_dim, embed_dim)
        self.Wk = nn.Linear(embed_dim, embed_dim)
        self.Wv = nn.Linear(embed_dim, embed_dim)

        # Output projection(s)
        if shared_out:
            self.Wo = nn.Linear(embed_dim, latent_dim)
        else:
            self.Wo = nn.ModuleList([nn.Linear(embed_dim, latent_dim) for _ in range(latent_dim)])

    def forward(self, Q, K, V, mask):
        """
        Apply multi-head attention.

        Args:
            Q: [B, P, E] - query tensor
            K: [B, S, E] - key tensor
            V: [B, S, E] - value tensor
            mask: [B, S] - boolean mask (True = valid)

        Returns:
            out: [B, P, latent_dim] - output coefficients
        """
        B, P, E = Q.shape
        _, S, _ = K.shape

        # Project and reshape Q, K, V into [B, h, P/S, k]
        Qp = self.Wq(Q).view(B, P, self.h, self.k).permute(0, 2, 1, 3)
        Kp = self.Wk(K).view(B, S, self.h, self.k).permute(0, 2, 1, 3)
        Vp = self.Wv(V).view(B, S, self.h, self.k).permute(0, 2, 1, 3)

        # Scaled dot-product attention
        scores = torch.einsum("bhpk,bhsk->bhps", Qp, Kp) / math.sqrt(self.k)
        m = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, S]
        scores = scores.masked_fill(~m, float("-inf"))
        A = F.softmax(scores, dim=-1)       # attention weights

        # Compute attention output
        C = torch.einsum("bhps,bhsk->bhpk", A, Vp)  # [B, h, P, k]
        C = C.permute(0, 2, 1, 3).contiguous().view(B, P, E)  # merge heads

        # Output projection
        if isinstance(self.Wo, nn.ModuleList):
            out = torch.stack([self.Wo[i](C[:, i]) for i in range(P)], dim=1)
        else:
            out = self.Wo(C)
        return out

class FLD(nn.Module):
    """
    Functional Latent Dynamics model with attention over time-channel inputs.

    Args:
        input_dim (int): Number of input channels C.
        latent_dim (int): Latent space size.
        num_heads (int): Number of attention heads.
        embed_dim (int): Embedding dimension.
        function (str): Basis type ('C', 'L', 'Q', 'S').
        residual_cycle (bool): Enable periodic baseline subtraction.
        cycle_length (int): Cycle length for periodic modeling.
    """
    def __init__(
        self, input_dim, latent_dim, num_heads, embed_dim, function,
        residual_cycle=False, cycle_length=24
    ):
        super().__init__()
        self.residual_cycle = residual_cycle
        self.cycle_length = cycle_length

        # Determine number of basis functions P
        if function == "C": P = 1
        elif function == "L": P = 2
        elif function == "Q": P = 3
        elif function == "S": P = 4
        else: raise ValueError(f"Unknown function {function}")
        self.P = P
        self.F = function
        self.E = embed_dim

        # Time embedding: linear projection + sinusoid
        self.time_linear = nn.Linear(1, embed_dim)
        # Channel embedding: learned per channel
        self.channel_embed = nn.Parameter(torch.randn(input_dim, embed_dim))
        # Attention layer
        self.attn = FLDAttention(embed_dim, latent_dim, num_heads)
        # Output decoder: latent → channels
        self.out = nn.Linear(latent_dim, input_dim)

    def _get_cycle(self, timesteps, X, M):
        """
        Compute and subtract per-phase cycle baseline.

        Args:
            timesteps: [B, T] - normalized timestamps
            X: [B, T, C] - input data
            M: [B, T, C] - valid entry mask

        Returns:
            c_in: [B, T, C] - baseline values per timestep
            c_base: [B, L_c, C] - mean cycle baseline
        """
        B, T, C = X.shape
        phases = torch.floor(timesteps).long().to(X.device) % self.cycle_length
        c_base = torch.zeros(B, self.cycle_length, C, device=X.device)

        for p in range(self.cycle_length):
            mask_p = (phases == p).unsqueeze(-1) & M.bool()
            sum_p = (X * mask_p).sum(dim=1)
            cnt_p = mask_p.sum(dim=1).clamp(min=1)
            c_base[:, p] = sum_p / cnt_p

        c_in = c_base.gather(1, phases.unsqueeze(-1).expand(-1, -1, C))
        return c_in, c_base

    def learn_time(self, tt):
        """
        Generate time embeddings via linear + sinusoidal transformation.

        Args:
            tt: [B, T] - normalized timestamps

        Returns:
            [B, T, E] time embeddings
        """
        E_lin = self.time_linear(tt.unsqueeze(-1))  # [B, T, E]
        return E_lin + torch.sin(E_lin)

    def forward(self, timesteps, X, M, y_times):
        """
        Forward pass of FLD model.

        Args:
            timesteps: [B, T] - input timestamps
            X: [B, T, C] - input values
            M: [B, T, C] - input mask
            y_times: [T_out] - future timestamps to predict

        Returns:
            Y_pred: [B, T_out, C] - predicted values
        """
        B, T, C = X.shape

        # Optional: subtract periodic residual baseline
        if self.residual_cycle:
            c_in, c_base = self._get_cycle(timesteps, X, M)
            X = X - c_in

        # Generate embeddings: time + channel
        Et = self.learn_time(timesteps).unsqueeze(2).expand(-1, -1, C, -1)  # [B, T, C, E]
        Ec = self.channel_embed.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)  # [B, T, C, E]
        E_all = Et + Ec  # [B, T, C, E]

        # Flatten time-channel to sequence [B, S, E]
        S = T * C
        K = E_all.view(B, S, self.E)
        V = K.clone()
        mask_flat = M.any(-1).view(B, T, 1).expand(-1, -1, C).reshape(B, S)

        # Learned query vector per batch (not trainable during inference)
        Q = nn.Parameter(torch.randn(B, self.P, self.E), requires_grad=False).to(X.device)

        # Apply attention
        coeffs = self.attn(Q, K, V, mask_flat)  # [B, P, latent_dim]

        # Decode: latent → channel predictions
        lat = coeffs.mean(dim=1)  # [B, latent_dim]
        Y_pred = self.out(lat).unsqueeze(1).expand(-1, y_times.size(1), -1)  # [B, T_out, C]

        # Optional: add periodic residual back for output time steps
        if self.residual_cycle:
            phases_out = torch.floor(y_times).long().to(X.device) % self.cycle_length
            c_fut = c_base.gather(1, phases_out.unsqueeze(-1).expand(-1, -1, C))
            Y_pred = Y_pred + c_fut

        return Y_pred
