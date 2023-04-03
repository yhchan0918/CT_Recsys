import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphSAGE, PairNorm


def triplet_margin_loss(anchor, positive, negative, margin):
    """
    Computes triplet margin loss.

    Args:
        anchor (torch.Tensor): Embedding tensor for anchor examples.
        positive (torch.Tensor): Embedding tensor for positive examples.
        negative (torch.Tensor): Embedding tensor for negative examples.
        margin (float): Margin hyperparameter for triplet loss.

    Returns:
        torch.Tensor: Triplet loss.
    """
    pos_dist = (anchor - positive).pow(2).sum(-1)
    neg_dist = (anchor - negative).pow(2).sum(-1)
    loss = F.relu(pos_dist - neg_dist + margin)
    return loss.mean()


class GraphSAGETripletRankingLoss(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGETripletRankingLoss, self).__init__()
        self.conv1 = GraphSAGE(in_channels, hidden_channels)
        self.conv2 = GraphSAGE(hidden_channels, out_channels)
        self.norm = PairNorm()

    def forward(self, x, pos_edge_index, neg_edge_index):
        x = self.conv1(x, pos_edge_index)
        x = F.relu(x)
        x = self.conv2(x, pos_edge_index)
        x = self.norm(x)
        anchor = x[pos_edge_index[0]]
        positive = x[pos_edge_index[1]]
        negative = x[neg_edge_index[1]]
        loss = triplet_margin_loss(anchor, positive, negative, margin=1.0)
        return loss


def train(model, optimizer, loader, device):
    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        x = batch.x.to(device)
        pos_edge_index = batch.edge_index.to(device)
        neg_edge_index = torch.randint(0, x.size(0), pos_edge_index.shape).to(device)
        loss = model(x, pos_edge_index, neg_edge_index)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(GraphSAGE, self).__init__()
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.projections = nn.ModuleList()

        # Add first projection layer for each node type
        self.projections.append(nn.Linear(in_channels[0], hidden_channels))
        self.projections.append(nn.Linear(in_channels[1], hidden_channels))

        # Add hidden layers
        for i in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
            self.projections.append(nn.Linear(hidden_channels, hidden_channels))

        # Add output layer
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x_dict, edge_index_dict):
        x_user = x_dict["user"]
        x_movie = x_dict["movie"]
        edge_index_user_pos = edge_index_dict["user_pos"]
        edge_index_user_neg = edge_index_dict["user_neg"]
        edge_index_movie_pos = edge_index_dict["movie_pos"]
        edge_index_movie_neg = edge_index_dict["movie_neg"]

        # User projection layer
        x_user = F.relu(self.projections[0](x_user))

        # Movie projection layer
        x_movie = F.relu(self.projections[1](x_movie))

        # Graph convolutional layers
        for i in range(self.num_layers - 1):
            x_user = F.relu(self.convs[i](x_user, edge_index_user_pos))
            x_user = self.batch_norms[i](x_user)
            x_user = F.dropout(x_user, p=0.5, training=self.training)
            x_user = self.projections[i + 1](x_user)

            x_movie = F.relu(self.convs[i](x_movie, edge_index_movie_pos))
            x_movie = self.batch_norms[i](x_movie)
            x_movie = F.dropout(x_movie, p=0.5, training=self.training)
            x_movie = self.projections[i + 1](x_movie)

        # Output layer
        x_user = self.convs[-1](x_user, edge_index_user_pos)
        x_movie = self.convs[-1](x_movie, edge_index_movie_pos)

        return x_user, x_movie
