# model_ablation.py
import torch
import torch.nn as nn
from models.fld_icc import FLD  # original attention-based FLD

class SimpleFLD(nn.Module):
    """
    Ablation: FLD without attention — just mean pooling + MLP decoder.
    """
    def __init__(self, input_dim, latent_dim, embed_dim, function="L", cycle_length=24, residual_cycle=False):
        super().__init__()
        self.residual_cycle = residual_cycle
        self.cycle_length = cycle_length
        self.input_dim = input_dim

        if function == "C": self.P = 1
        elif function == "L": self.P = 2
        elif function == "Q": self.P = 3
        elif function == "S": self.P = 4
        else: raise ValueError(f"Unknown function {function}")
        self.E = embed_dim

        self.time_linear = nn.Linear(1, embed_dim)
        self.channel_embed = nn.Parameter(torch.randn(input_dim, embed_dim))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, input_dim)
        )

    def learn_time(self, tt):
        E_lin = self.time_linear(tt.unsqueeze(-1))
        return E_lin + torch.sin(E_lin)

    def _get_cycle(self, timesteps, X, M):
        B, T, C = X.shape
        phases = torch.floor(timesteps).long().to(X.device) % self.cycle_length
        c_base = torch.zeros(B, self.cycle_length, C, device=X.device)
        for p in range(self.cycle_length):
            mask_p = (phases == p).unsqueeze(-1) & M.bool()
            sum_p = (X * mask_p).sum(dim=1)
            cnt_p = mask_p.sum(dim=1).clamp(min=1)
            c_base[:, p] = sum_p / cnt_p
        c_in = c_base.gather(1, phases.unsqueeze(-1).expand(-1,-1,C))
        return c_in, c_base

    def forward(self, timesteps, X, M, y_times):
        B, T, C = X.shape

        if self.residual_cycle:
            c_in, c_base = self._get_cycle(timesteps, X, M)
            X = X - c_in

        Et = self.learn_time(timesteps).unsqueeze(2).expand(-1,-1,C,-1)
        Ec = self.channel_embed.unsqueeze(0).unsqueeze(0).expand(B,T,-1,-1)
        E_all = Et + Ec  # [B, T, C, E]

        E_seq = E_all.view(B, T * C, self.E).permute(0, 2, 1)  # [B, E, T*C]
        pooled = self.pool(E_seq).squeeze(-1)  # [B, E]

        Y_pred = self.decoder(pooled).unsqueeze(1).expand(-1, y_times.size(1), -1)

        if self.residual_cycle:
            phases_out = torch.floor(y_times).long().to(X.device) % self.cycle_length
            c_fut = c_base.gather(1, phases_out.unsqueeze(-1).expand(-1, -1, C))
            Y_pred = Y_pred + c_fut

        return Y_pred
