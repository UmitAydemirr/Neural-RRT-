import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, BatchNorm

class NeuralPlannerGAT(nn.Module):
    def __init__(self, in_channels=5, hidden_channels=64, heads=4, dropout=0.1):
        super().__init__()
        self.dropout_rate = dropout
        self.concat_dim = hidden_channels * heads

        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, concat=True, dropout=dropout)
        self.bn1 = BatchNorm(self.concat_dim)

        self.conv2 = GATv2Conv(self.concat_dim, hidden_channels, heads=heads, concat=True, dropout=dropout, residual=True)
        self.bn2 = BatchNorm(self.concat_dim)

        self.out_heads = 2
        self.out_dim = hidden_channels * self.out_heads
        self.conv3 = GATv2Conv(self.concat_dim, hidden_channels, heads=self.out_heads, concat=True, dropout=dropout, residual=True)
        self.bn3 = BatchNorm(self.out_dim)

        final_input_dim = self.out_dim + in_channels
        self.classifier = nn.Sequential(
            nn.Linear(final_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x, edge_index):
        x_raw = x

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.elu(x)

        x_combined = torch.cat([x, x_raw], dim=1)
        out = self.classifier(x_combined)
        return out.squeeze(-1)