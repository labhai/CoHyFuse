import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class ConditionHypergraph(nn.Module):
    """H[j, i] is the membership of ROI j in the hyperedge anchored at i."""

    def __init__(self, k_per_condition: Sequence[int]):
        super().__init__()
        self.k_per_condition = tuple(k_per_condition)
        initial = math.log(math.expm1(1.0))
        self.raw_temperature = nn.Parameter(torch.full((len(k_per_condition),), initial))

    @property
    def temperature(self) -> torch.Tensor:
        return F.softplus(self.raw_temperature) + 1e-6

    def forward(self, nodes: torch.Tensor, condition: int) -> torch.Tensor:
        batch, rois, _ = nodes.shape
        k = self.k_per_condition[condition]
        squared_norm = nodes.square().sum(dim=-1, keepdim=True)
        distances = squared_norm + squared_norm.transpose(1, 2) - 2 * nodes @ nodes.transpose(1, 2)
        similarity = -distances.clamp_min(0)
        diagonal = torch.eye(rois, dtype=torch.bool, device=nodes.device)
        values, neighbors = similarity.masked_fill(diagonal, -torch.inf).topk(k, dim=-1)
        weights = k * (values / self.temperature[condition]).softmax(dim=-1)
        anchor_rows = nodes.new_zeros(batch, rois, rois).scatter(2, neighbors, weights)
        return anchor_rows.transpose(1, 2) + diagonal.to(nodes.dtype)


class HypergraphConvolution(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def propagate(incidence: torch.Tensor, nodes: torch.Tensor) -> torch.Tensor:
        vertex_degree = incidence.sum(dim=2).clamp_min(1e-8)
        edge_degree = incidence.sum(dim=1).clamp_min(1e-8)
        scale = vertex_degree.rsqrt().unsqueeze(-1)
        edges = incidence.transpose(1, 2) @ (scale * nodes)
        edges = edges / edge_degree.unsqueeze(-1)
        return scale * (incidence @ edges)

    def forward(self, incidence: torch.Tensor, nodes: torch.Tensor) -> torch.Tensor:
        return self.dropout(F.relu(self.linear(self.propagate(incidence, nodes))))


class CoHyFuse(nn.Module):
    """Condition FCs (B,Q,V,V) and whole-session FC vectors (B,V*(V-1)/2)."""

    def __init__(
        self,
        num_rois: int = 100,
        k_per_condition: Sequence[int] = (50, 45, 5),
        hidden_dim: int = 64,
        fc_hidden_dim: int = 128,
        heads: int = 4,
        dropout: float = 0.2,
        num_classes: int | None = None,
    ):
        super().__init__()
        if num_rois < 2 or not k_per_condition:
            raise ValueError("At least two ROIs and one condition are required.")
        if any(not isinstance(k, int) or not 1 <= k < num_rois for k in k_per_condition):
            raise ValueError("Each K must be an integer in [1, num_rois - 1].")
        if heads < 1 or hidden_dim < 1 or hidden_dim % heads or fc_hidden_dim < 1:
            raise ValueError(
                "Hidden sizes must be positive; hidden_dim must be divisible by heads."
            )
        if num_classes is not None and num_classes < 2:
            raise ValueError("Classification requires at least two classes.")
        self.num_rois = num_rois
        self.num_conditions = len(k_per_condition)
        self.num_classes = num_classes
        self.fc_dim = num_rois * (num_rois - 1) // 2
        self.fc_encoder = nn.Sequential(
            nn.Linear(num_rois, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.hypergraph = ConditionHypergraph(k_per_condition)
        self.hgnn = HypergraphConvolution(hidden_dim, dropout)
        self.pool_score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.fusion = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=2 * hidden_dim,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.whole_fc = nn.Sequential(
            nn.Linear(self.fc_dim, fc_hidden_dim),
            nn.LayerNorm(fc_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_dim, hidden_dim),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_dim, fc_hidden_dim),
            nn.LayerNorm(fc_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_dim, num_classes or 1),
        )

    def forward(self, condition_fc: torch.Tensor, whole_fc: torch.Tensor) -> torch.Tensor:
        expected = (self.num_conditions, self.num_rois, self.num_rois)
        if condition_fc.ndim != 4 or condition_fc.shape[1:] != expected:
            raise ValueError(f"condition_fc must have shape (B,{expected}).")
        if whole_fc.shape != (condition_fc.shape[0], self.fc_dim):
            raise ValueError(f"whole_fc must have shape (B,{self.fc_dim}).")
        conditions = []
        for q in range(self.num_conditions):
            nodes = self.fc_encoder(condition_fc[:, q])
            incidence = self.hypergraph(nodes, q)
            nodes = self.hgnn(incidence, nodes)
            attention = self.pool_score(nodes).softmax(dim=1)
            conditions.append((attention * nodes).sum(dim=1))
        condition_embedding = self.fusion(torch.stack(conditions, dim=1)).mean(dim=1)
        embedding = torch.cat((condition_embedding, self.whole_fc(whole_fc)), dim=-1)
        prediction = self.head(embedding)
        return prediction.squeeze(-1) if self.num_classes is None else prediction
