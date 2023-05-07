import torch
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv, to_hetero
from torchmetrics.functional import pairwise_cosine_similarity


class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        # Mean pooling, by default
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


class Model(torch.nn.Module):
    def __init__(self, hidden_channels, data):
        super().__init__()
        self.encoder = GNNEncoder(hidden_channels, hidden_channels)
        self.encoder = to_hetero(self.encoder, data.metadata(), aggr="sum")

    def forward(self, x_dict, edge_index_dict):
        # Node embedding here
        return self.encoder(x_dict, edge_index_dict)

    def inference(self, x_dict, edge_index_dict):
        return self.encoder(x_dict, edge_index_dict)


def prepare_data_loader(
    data,
    batch_size,
    num_neighbours,
):
    neighbourhood_sampling_loader = NeighborLoader(
        data,
        batch_size=batch_size,
        num_neighbors=num_neighbours,
        shuffle=True,
        input_nodes=("listing", None),
    )
    return neighbourhood_sampling_loader


def triplet_ranking_loss(emb, rating, edge_index, batch_size, margin_constant=0.13):
    # Compute pairwise cosine similarity between user embedding and listing embedding
    cosine_similarity_matrix = pairwise_cosine_similarity(emb["user"], emb["listing"][:batch_size])
    # Sample adjacency matrix (binary) represented by two tensors
    src_tensor = edge_index[0]
    dst_tensor = edge_index[1]
    max_listing_idx = dst_tensor.max()
    # Number of nodes
    n_nodes = max(src_tensor.max(), max_listing_idx) + 1

    # Convert adjacency matrix to dictionary
    adj_dict = {}
    edge_label_dict = {}
    for i in range(n_nodes):
        adj_dict[i] = list(dst_tensor[src_tensor == i].numpy())
        edge_label_dict[i] = list(rating[src_tensor == i].numpy())

    def exclude_elem(a, b):
        mask = torch.ones_like(b, dtype=torch.bool)
        mask[a] = 0
        return torch.masked_select(b, mask)

    listing_indices = torch.arange(max_listing_idx + 1)
    hardest_pos_sims = []
    hardest_neg_sims = []
    ratings = []
    for i in adj_dict:
        pos_idx = torch.tensor(adj_dict[i])
        neg_idx = exclude_elem(pos_idx, listing_indices)
        # If there are no negative examples, skip this user
        if len(neg_idx) == 0:
            continue

        # Retrieve distance between the positive and negative examples
        pos_sim = cosine_similarity_matrix[i, pos_idx]
        neg_sim = cosine_similarity_matrix[i, neg_idx]
        # Select the hardest negative example and hardest postive example
        hardest_pos_idx = torch.argmin(pos_sim)
        hardest_pos_sim = pos_sim[hardest_pos_idx]
        pos_rating = torch.tensor(edge_label_dict[i])[hardest_pos_idx]
        hardest_neg_idx = torch.argmax(neg_sim)
        hardest_neg_sim = neg_sim[hardest_neg_idx]
        hardest_pos_sims.append(hardest_pos_sim)
        hardest_neg_sims.append(hardest_neg_sim)
        ratings.append(pos_rating)

    hardest_pos_sims = torch.stack(hardest_pos_sims, dim=0)
    hardest_neg_sims = torch.stack(hardest_neg_sims, dim=0)
    ratings = torch.stack(ratings, dim=0)
    m = ratings * margin_constant

    # Combine most disimilar s(a, p) and most similar s(a, n) into final triplet loss
    triplet_loss = torch.maximum(torch.zeros(m.size()), -hardest_pos_sims + hardest_neg_sims + m)
    return triplet_loss.mean()


def train(model, optimizer, train_loader, device):
    model.train(True)
    total_examples = total_loss = 0
    # Why using mini-batch gradient descent
    # Update NN multiple times every epoch, Make more precise update to the parameters by calculating the average loss in each step
    # Reduce overall training time and num of required epochs for reaching convergence, computational efficiency
    for batch in train_loader:
        batch = batch.to(device)
        # Zero gradients for every batch
        optimizer.zero_grad()
        # Make predictions for this batch
        emb = model(batch.x_dict, batch.edge_index_dict)
        batch_size = batch["listing"].batch_size
        rating = batch["user", "listing"].edge_label
        edge_index = batch["user", "listing"].edge_index
        loss = triplet_ranking_loss(emb, rating, edge_index, batch_size)
        # Compute the loss and its gradients
        loss.backward()
        # Adjust learning weights
        optimizer.step()
        total_loss += float(loss) * batch_size
        total_examples += batch_size

    train_loss = total_loss / total_examples
    return train_loss


@torch.no_grad()
def test(test_loader, device, model):
    model.eval()
    total_examples = total_loss = 0
    for batch in test_loader:
        batch = batch.to(device)
        # Make predictions for this batch
        emb = model(batch.x_dict, batch.edge_index_dict)
        batch_size = batch["listing"].batch_size
        rating = batch["user", "listing"].edge_label
        edge_index = batch["user", "listing"].edge_index
        loss = triplet_ranking_loss(emb, rating, edge_index, batch_size)
        total_loss += float(loss) * batch_size
        total_examples += batch_size

    test_loss = total_loss / total_examples
    return test_loss


def load_model(path, hidden_channels, data):
    model = Model(hidden_channels=hidden_channels, data=data).to("cpu")
    model.load_state_dict(torch.load(path))
    return model
