import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import TripletMarginLoss
from torch.nn.utils import clip_grad_norm_
from torch.nn.functional import normalize
from torch.utils.data import Dataset
from torch_geometric.data import Data, HeteroDataLoader
from torch_geometric.nn import GraphSAGEConv


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
        return len(self.user_movie_interactions)

    def __getitem__(self, idx):
        user, movie, label = self.user_movie_interactions[idx], self.labels[idx]
        user_node = self.user_nodes[user]
        movie_node = self.movie_nodes[movie]
        movie_genres = self.movie_genre_categorizations[movie]
        return user_node, movie_node, movie_genres, label


class RecommendationModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.convs.append(GraphSAGEConv(input_dim, hidden_dim))
        for i in range(1, num_layers):
            self.convs.append(GraphSAGEConv(hidden_dim, hidden_dim))
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x_dict, edge_index_dict):
        x_dict = {k: normalize(v, dim=-1) for k, v in x_dict.items()}
        for i in range(self.num_layers):
            x_dict = {
                k: F.relu(conv(x_dict[k], edge_index_dict[(k, "followed_by")]))
                for k, conv in enumerate(self.convs)
            }
            x_dict = {k: normalize(v, dim=-1) for k, v in x_dict.items()}
        user_emb = x_dict[0]
        movie_emb = x_dict[1]
        movie_genre_emb = x_dict[2]
        return user_emb, movie_emb, movie_genre_emb


class TripletMarginLossModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss_fn = nn.TripletMarginLoss(margin=1.0, p=2)

    def forward(self, user_emb, pos_movie_emb, neg_movie_emb):
        pos_dist = torch.cdist(user_emb, pos_movie_emb)
        neg_dist = torch.cdist(user_emb, neg_movie_emb)
        loss = self.loss_fn(pos_dist, neg_dist, torch.ones_like(pos_dist))
        return loss


def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch_idx, data in enumerate(train_loader):
        user_emb, pos_movie_emb, movie_genre_emb, neg_movie_emb = [d.to(device) for d in data]
        optimizer.zero_grad()
        user_emb, pos_movie_emb, neg_movie_emb = model(user_emb, pos_movie_emb, neg_movie_emb)
        loss = TripletMarginLossModel(model)(user_emb, pos_movie_emb, neg_movie_emb)
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


# Define your Heterogeneous graph using PyTorch Geometric's HeteroData class
data = HeteroData()
data["user_nodes"] = torch.randn(100, 16)
data["movie_nodes"] = torch.randn(500, 16)
data.add_edge_index("followed_by", torch.randint(0, 100, (1000,)), torch.randint(0, 100, (1000,)))
data.add_edge_index("watched", torch.randint(0, 100, (5000,)), torch.randint(0, 500, (5000,)))
data.add_edge_index("has_genre", torch.randint(0, 500, (5000,)), torch.randint(0, 20, (5000,)))

# Define your model
model = RecommendationModel(input_dim=16, hidden_dim=32, num_layers=2)
model = TripletMarginLossModel(model)

# Define your training data loader
train_dataset = MovieDataset(
    data["user_nodes"],
    data["movie_nodes"],
    data["watched"],
    data["has_genre"],
    torch.randn(5000, 1),
)
train_loader = HeteroDataLoader(train_dataset, batch_size=64, shuffle=True)

# Train your model
optimizer = Adam(model.parameters(), lr=0.01)
for epoch in range(10):
    loss = train(model, train_loader, optimizer, "cpu")
    print(f"Epoch {epoch+1} | Loss: {loss:.4f}")
