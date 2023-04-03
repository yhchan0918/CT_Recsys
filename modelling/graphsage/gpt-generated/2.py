import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import HeteroGraphSage


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


class RecommendationModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(RecommendationModel, self).__init__()
        self.conv = HeteroGraphSage(input_dim, hidden_dim, num_layers)
        self.fc = torch.nn.Linear(hidden_dim, 1)

    def forward(
        self, user_nodes, movie_nodes, user_movie_interactions, movie_genre_categorizations
    ):
        user_features = self.conv(user_nodes, user_movie_interactions, None, "user", "user_movie")
        movie_features = self.conv(
            movie_nodes, None, movie_genre_categorizations, "movie", "movie_genre"
        )

        # Compute pairwise distances between all user and movie features
        dist_matrix = torch.cdist(user_features, movie_features, p=2)
        pos_pairs = []
        neg_pairs = []

        # Iterate over each user and their interactions
        for i, (user_node, movie_node) in enumerate(user_movie_interactions):
            pos_idx = torch.where(movie_nodes == movie_node)[0]
            neg_idx = torch.where(labels != labels[i])[0]

            # If there are no negative examples, skip this interaction
            if len(neg_idx) == 0:
                continue

            # Compute distance between the positive and negative examples
            pos_dist = dist_matrix[i, pos_idx]
            neg_dist = dist_matrix[i, neg_idx]

            # Select the hardest negative example
            hardest_neg_idx = torch.argmin(neg_dist)
            hardest_neg_dist = neg_dist[hardest_neg_idx]
            pos_pairs.append(pos_dist)
            neg_pairs.append(hardest_neg_dist)

        if len(pos_pairs) > 0:
            pos_pairs = torch.cat(pos_pairs, dim=0)
            neg_pairs = torch.cat(neg_pairs, dim=0)
            loss = F.triplet_margin_loss(pos_pairs, neg_pairs, margin=1.0)
        else:
            loss = torch.tensor(0.0)

        return self.fc(user_features), self.fc(movie_features), loss


# Define hyperparameters
input_dim = 32
hidden_dim = 16
num_layers = 2
lr = 0.01
batch_size = 32
num_epochs = 10

# Load data and create data
