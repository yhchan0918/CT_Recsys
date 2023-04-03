import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class HeteroGraphSAGE(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_user_features,
        num_item_features,
        num_rating_features,
    ):
        super(HeteroGraphSAGE, self).__init__()

        self.user_proj = torch.nn.Linear(num_user_features, hidden_channels)
        self.item_proj = torch.nn.Linear(num_item_features, hidden_channels)
        self.rating_proj = torch.nn.Linear(num_rating_features, hidden_channels)

        self.conv1 = SAGEConv((in_channels, hidden_channels), hidden_channels)
        self.conv2 = SAGEConv((hidden_channels, hidden_channels), out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.user_proj.reset_parameters()
        self.item_proj.reset_parameters()
        self.rating_proj.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(
        self,
        x_user,
        x_item,
        x_rating,
        edge_index_user_item,
        edge_weight_user_item,
        edge_index_user_rating,
        edge_weight_user_rating,
    ):
        user_embeddings = F.relu(self.user_proj(x_user))
        item_embeddings = F.relu(self.item_proj(x_item))
        rating_embeddings = F.relu(self.rating_proj(x_rating))

        x = torch.cat([user_embeddings, item_embeddings, rating_embeddings], dim=0)
        edge_index = torch.cat([edge_index_user_item, edge_index_user_rating], dim=1)
        edge_weight = torch.cat([edge_weight_user_item, edge_weight_user_rating], dim=0)

        x = self.conv1((x, x), edge_index, edge_weight)
        x = F.relu(x)
        x = self.conv2((x, x), edge_index, edge_weight)

        return x[: len(user_embeddings)]
