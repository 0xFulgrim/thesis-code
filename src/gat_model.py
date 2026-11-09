"""
gat_model.py — Graph Attention Network baseline for money mule detection.

Approach:
  - Same static graph as GraphSAGE (accounts=nodes, transactions=edges,
    same StandardScaler-normalised node features).
  - GATv2 (Brody et al. 2021): dynamic multi-head attention over neighbours.
    Structurally meaningful for mule detection — a hub account learns to
    attend more to high-risk counterparties than routine ones.
  - Two GATv2Conv layers (4 heads concat → 1 head) + BatchNorm + linear head.
  - Mini-batch training via NeighborLoader.
  - Validation split for early stopping — test set never touched during training.

The training pipeline (focal loss, loaders, early stopping, evaluation) is
shared with graphsage_model.py and lives in static_gnn.py.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

from static_gnn import build_parser, run

# ── Model ─────────────────────────────────────────────────────────────────────

class GAT(torch.nn.Module):
    """
    Two-layer GATv2 with BatchNorm and a linear classification head.

    Architecture:
      GATv2Conv(in, hidden, heads=4, concat=True)  → hidden*4 → BN → ELU → Dropout
      GATv2Conv(hidden*4, hidden, heads=1, concat=False) → BN → ELU → Dropout
      Linear(hidden → 2)

    dropout=0.0 inside GATv2Conv — feature dropout is handled by F.dropout.
    Stacking both would compound to ~51% effective dropout through layer 1,
    killing minority class gradients.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.conv1   = GATv2Conv(in_channels,             hidden_channels,
                                 heads=heads, dropout=0.0, concat=True)
        self.conv2   = GATv2Conv(hidden_channels * heads, hidden_channels,
                                 heads=1,     dropout=0.0, concat=False)
        self.bn1     = torch.nn.BatchNorm1d(hidden_channels * heads)
        self.bn2     = torch.nn.BatchNorm1d(hidden_channels)
        self.lin     = torch.nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.bn2(self.conv2(x, edge_index)))
        return self.lin(x)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = build_parser("GATv2 mule detector", hidden_channels=64, lr=5e-4)
    parser.add_argument("--heads", type=int, default=4)
    args = parser.parse_args()

    run(
        args,
        model_name="GATv2",
        file_prefix="gat",
        build_model=lambda in_channels: GAT(
            in_channels=in_channels,
            hidden_channels=args.hidden_channels,
            out_channels=2,
            heads=args.heads,
            dropout=args.dropout,
        ),
    )


if __name__ == "__main__":
    main()
