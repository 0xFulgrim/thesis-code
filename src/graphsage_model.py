"""
graphsage_model.py — Static inductive GNN baseline for money mule detection.

Approach:
  - Accounts are nodes; each transaction is a directed edge.
  - Node features are the same aggregate statistics as the LightGBM baseline,
    normalised with StandardScaler (applied inside build_static_graph).
    This isolates the effect of adding graph structure vs. tabular features alone.
  - GraphSAGE (Hamilton et al. 2017): inductive, 2-hop sampled aggregation.
    Two SAGEConv layers + BatchNorm + linear classification head.
  - Mini-batch training via NeighborLoader.
  - Validation split for early stopping — test set never touched during training.

The training pipeline (focal loss, loaders, early stopping, evaluation) is
shared with gat_model.py and lives in static_gnn.py.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from static_gnn import build_parser, run

# ── Model ─────────────────────────────────────────────────────────────────────

class GraphSAGE(torch.nn.Module):
    """
    Two-layer GraphSAGE with BatchNorm and a linear classification head.

    Architecture:
      SAGEConv(in → hidden) → BatchNorm → ReLU → Dropout
      SAGEConv(hidden → hidden) → BatchNorm → ReLU → Dropout
      Linear(hidden → 2)

    BatchNorm is placed after the convolution and before the activation,
    following standard practice for GNNs with high-variance node degrees.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.conv1 = SAGEConv(in_channels,     hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.bn1   = torch.nn.BatchNorm1d(hidden_channels)
        self.bn2   = torch.nn.BatchNorm1d(hidden_channels)
        self.lin   = torch.nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.bn1(self.conv1(x, edge_index)).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.bn2(self.conv2(x, edge_index)).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin(x)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = build_parser(
        "GraphSAGE mule detector", hidden_channels=128, lr=1e-3,
    ).parse_args()

    run(
        args,
        model_name="GraphSAGE",
        file_prefix="graphsage",
        build_model=lambda in_channels: GraphSAGE(
            in_channels=in_channels,
            hidden_channels=args.hidden_channels,
            out_channels=2,
            dropout=args.dropout,
        ),
    )


if __name__ == "__main__":
    main()
