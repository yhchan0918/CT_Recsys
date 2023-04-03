import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader


class HeteroGraphSage(nn.Module):
    def __init__(self, node_type_dims, edge_type_dims, hidden_dim, num_neighbors, output_dim):
        super(HeteroGraphSage, self).__init__()
        self.node_type_dims = node_type_dims
        self.edge_type_dims = edge_type_dims
        self.hidden_dim = hidden_dim
        self.num_neighbors = num_neighbors
        self.output_dim = output_dim

        # Define linear layers for each node and edge type
        self.node_linear_layers = nn.ModuleDict()
        self.edge_linear_layers = nn.ModuleDict()
        for node_type, node_dim in node_type_dims.items():
            self.node_linear_layers[node_type] = nn.Linear(node_dim + hidden_dim, hidden_dim)
        for edge_type, edge_dim in edge_type_dims.items():
            self.edge_linear_layers[edge_type] = nn.Linear(edge_dim + hidden_dim, hidden_dim)

        # Define final linear layer for output
        self.output_linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, node_features, neighbor_features, edge_features, node_type, edge_type):
        # Compute mean of neighbor embeddings
        h_agg = torch.mean(neighbor_features, dim=1)

        # Concatenate node and edge features with aggregated neighbor embeddings
        h_node_concat = torch.cat([node_features, h_agg], dim=-1)
        h_edge_concat = torch.cat([edge_features, h_agg], dim=-1)

        # Pass through linear layers for node and edge types
        h_node_hidden = F.relu(self.node_linear_layers[node_type](h_node_concat))
        h_edge_hidden = F.relu(self.edge_linear_layers[edge_type](h_edge_concat))

        # Combine node and edge embeddings
        h_combined = h_node_hidden + h_edge_hidden

        # Pass through final linear layer for output
        h_output = self.output_linear(h_combined)

        return h_output


class HingeLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(HingeLoss, self).__init__()
        self.margin = margin

    def forward(self, logits, labels):
        # Compute hinge loss
        loss = self.margin - labels * logits
        loss = torch.clamp(loss, min=0.0)
        loss = torch.mean(loss)

        return loss


class MovieDataset(Dataset):
    def __init__(
        self, user_nodes, movie_nodes, user_movie_interactions, movie_genre_categorizations, labels
    ):
        self.user_nodes = user_nodes
        self.movie_nodes = movie_nodes
        self.user_movie_interactions = user_movie_interactions
        self.movie_genre_categorizations = movie_genre_categorizations
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        user_node = self.user_nodes[idx]
        movie_node = self.movie_nodes[idx]
        user_movie_interaction = self.user_movie_interactions[idx]
        movie_genre_categorization = self.movie_genre_categorizations[idx]
        label = self.labels[idx]

        return user_node, movie_node, user_movie_interaction, movie_genre_categorization, label


def train(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    for (
        user_nodes,
        movie_nodes,
        user_movie_interactions,
        movie_genre_categorizations,
        labels,
    ) in dataloader:
        optimizer.zero_grad()

        # Forward pass
        user_features = model(user_nodes, user_movie_interactions, None, "user", "user_movie")
        movie_features = model(
            movie_nodes, None, movie_genre_categorizations, "movie", "movie_genre"
        )
        logits = torch.sum(user_features * movie_features, dim=-1)
        loss = criterion(logits, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)

    return total_loss / len(dataloader.dataset)
