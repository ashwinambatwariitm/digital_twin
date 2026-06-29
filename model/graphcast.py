"""
Mini-GraphCast
──────────────
GraphCast-inspired GNN for India climate forecasting.
Encode → Process (message passing) → Decode

Designed to run on RTX 3050 Laptop (4GB VRAM) with:
  - ~500 nodes (0.5° grid)
  - hidden_dim = 128
  - 6 message passing layers

Architecture follows DeepMind's GraphCast paper (Lam et al. 2023)
but at 1/100th the scale — appropriate for 1-year training data.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
import logging

log = logging.getLogger(__name__)


# ── Message Passing Layer ─────────────────────────────────────────────────────

class GraphCastLayer(MessagePassing):
    """
    One round of message passing.

    For each node:
      1. Collect messages from all neighbors (edge MLP)
      2. Aggregate messages (mean)
      3. Update node state (node MLP)
      4. Add residual connection
    """

    def __init__(self, hidden_dim: int, edge_dim: int = 4, dropout: float = 0.1):
        super().__init__(aggr="mean")

        # Edge network: combines sender + receiver features + edge geometry
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # Node update network
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h          : (N, hidden_dim) node embeddings
            edge_index : (2, E)
            edge_attr  : (E, 4)

        Returns:
            h_new : (N, hidden_dim) updated embeddings
        """
        h_new = self.propagate(edge_index, h=h, edge_attr=edge_attr)
        return self.norm(h + h_new)   # residual connection

    def message(self, h_i: torch.Tensor, h_j: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        """Compute message from node j → node i."""
        msg = torch.cat([h_i, h_j, edge_attr], dim=-1)
        return self.edge_mlp(msg)

    def update(self, aggr_out: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Update node embedding from aggregated messages."""
        node_in = torch.cat([h, aggr_out], dim=-1)
        return self.node_mlp(node_in)


# ── Full Mini-GraphCast ───────────────────────────────────────────────────────

class MiniGraphCast(nn.Module):
    """
    Full GraphCast-inspired model.

    Input  : (B, seq_len, N, C)  — batch of sequences
    Output : (B, pred_steps, N, 3) — forecast (rainfall, tmax, tmin)

    The model processes time via a learned temporal aggregation
    before the graph encoder (keeps memory low on laptop GPU).
    """

    def __init__(
        self,
        node_features:    int = 6,    # IMD(3) + INSAT(3)
        hidden_dim:       int = 128,
        n_process_layers: int = 6,
        seq_len:          int = 7,
        pred_steps:       int = 3,
        n_targets:        int = 3,    # rainfall, tmax, tmin
        edge_dim:         int = 4,
        dropout:          float = 0.1,
    ):
        super().__init__()
        self.seq_len          = seq_len
        self.pred_steps       = pred_steps
        self.n_targets        = n_targets
        self.hidden_dim       = hidden_dim
        self.n_process_layers = n_process_layers

        # ── Temporal Encoder ───────────────────────────────────
        # Compress seq_len timesteps into one context vector per node
        self.temporal_encoder = nn.Sequential(
            nn.Linear(node_features * seq_len, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # ── Graph Encoder ──────────────────────────────────────
        # Map node features → latent space
        self.node_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Edge encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, edge_dim),   # keep edge_dim for layers
        )

        # ── Processor: N rounds of message passing ─────────────
        self.processor = nn.ModuleList([
            GraphCastLayer(hidden_dim, edge_dim, dropout)
            for _ in range(n_process_layers)
        ])

        # ── Decoder ────────────────────────────────────────────
        # Produce pred_steps × n_targets per node
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, n_targets * pred_steps),
        )

        self._init_weights()
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        log.info(f"MiniGraphCast: {n_params:,} trainable parameters")

    def forward(
        self,
        x:          torch.Tensor,   # (B, seq_len, N, C)
        edge_index: torch.Tensor,   # (2, E)
        edge_attr:  torch.Tensor,   # (E, 4)
    ) -> torch.Tensor:
        """
        Returns:
            predictions : (B, pred_steps, N, n_targets)
        """
        B, S, N, C = x.shape

        # ── Temporal aggregation ──────────────────────────────
        # Reshape: (B, N, S*C) → temporal_encoder → (B, N, hidden)
        x_flat = x.permute(0, 2, 1, 3).reshape(B, N, S * C)
        h = self.temporal_encoder(x_flat)         # (B, N, hidden)

        # ── Process each item in batch ────────────────────────
        # (Graph ops work on single graphs; loop over batch)
        # For small graphs this is fast enough on laptop
        outputs = []
        for b in range(B):
            h_b    = self.node_encoder(h[b])      # (N, hidden)
            e_attr = self.edge_encoder(edge_attr) # (E, edge_dim)

            # Message passing rounds
            for layer in self.processor:
                h_b = layer(h_b, edge_index, e_attr)

            # Decode: (N, pred_steps * n_targets)
            out_b = self.decoder(h_b)
            out_b = out_b.reshape(N, self.pred_steps, self.n_targets)
            out_b = out_b.permute(1, 0, 2)        # (pred_steps, N, n_targets)
            outputs.append(out_b)

        return torch.stack(outputs, dim=0)         # (B, pred_steps, N, n_targets)

    def predict_autoregressive(
        self,
        x:          torch.Tensor,   # (1, seq_len, N, C) — single sample
        edge_index: torch.Tensor,
        edge_attr:  torch.Tensor,
        n_steps:    int = 3,
    ) -> torch.Tensor:
        """
        Autoregressive rollout: use each prediction as the next input.
        Useful for longer-range forecasts.

        Returns: (n_steps, N, n_targets)
        """
        self.eval()
        predictions = []

        with torch.no_grad():
            current_x = x.clone()   # (1, seq_len, N, C)

            for step in range(n_steps):
                pred = self.forward(current_x, edge_index, edge_attr)
                # pred: (1, pred_steps, N, n_targets)
                pred_step = pred[0, 0]  # take first step: (N, n_targets)
                predictions.append(pred_step)

                # Slide window: drop oldest timestep, append prediction
                # Update only the target channels (0, 1, 2)
                new_frame = current_x[0, -1].clone()   # (N, C)
                new_frame[:, :3] = pred_step            # update rain, tmax, tmin
                current_x = torch.cat([
                    current_x[:, 1:],                   # drop oldest
                    new_frame.unsqueeze(0).unsqueeze(0) # add new
                ], dim=1)

        return torch.stack(predictions, dim=0)   # (n_steps, N, n_targets)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)