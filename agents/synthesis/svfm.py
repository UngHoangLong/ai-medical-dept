import torch
import torch.nn as nn


class SVFMSynthesis(nn.Module):
    """
    Structured Vision-Findings Merger.
    Fuses hidden states from 3 specialist agents via cross-attention,
    producing a unified embedding for retrieval and multi-task classification.
    """

    def __init__(self, d_model: int = 2048, d_fused: int = 512, n_agents: int = 3):
        super().__init__()
        self.proj = nn.ModuleList([nn.Linear(d_model, d_fused) for _ in range(n_agents)])
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_fused, nhead=8, batch_first=True, dropout=0.1),
            num_layers=2,
        )
        self.norm = nn.LayerNorm(d_fused)

    def forward(self, hidden_states: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            hidden_states: list of [B, D_model] tensors, one per agent
        Returns:
            unified_embedding: [B, D_fused]
        """
        tokens = torch.stack(
            [proj(h).unsqueeze(1) for proj, h in zip(self.proj, hidden_states)],
            dim=1,
        ).squeeze(2)  # [B, n_agents, D_fused]

        fused = self.transformer(tokens)           # [B, n_agents, D_fused]
        unified = self.norm(fused.mean(dim=1))     # [B, D_fused]
        return unified
