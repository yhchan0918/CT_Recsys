import torch
import torch.nn.functional as F
from torch.nn import Linear

import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv, to_hetero
import json
import pandas as pd


from prepare_graph import *
from evaluate import *


def write_dict_into_json(dictionary, filename):
    with open(filename, "w") as write_file:
        json.dump(dictionary, write_file, indent=4)


def weighted_mse_loss(pred, target, weight=None):
    weight = 1.0 if weight is None else weight[target].to(pred.dtype)
    return (weight * (pred - target.to(pred.dtype)).pow(2)).mean()


class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_dict["user"][row], z_dict["listing"][col]], dim=-1)

        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)


class Model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.encoder = GNNEncoder(hidden_channels, hidden_channels)
        self.encoder = to_hetero(self.encoder, train_data.metadata(), aggr="sum")
        self.decoder = EdgeDecoder(hidden_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        # Node embedding here
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)

    def inference(self, x_dict, edge_index_dict):
        return self.encoder(x_dict, edge_index_dict)


def train():
    model.train()
    optimizer.zero_grad()
    pred = model(
        train_data.x_dict,
        train_data.edge_index_dict,
        train_data["user", "listing"].edge_label_index,
    )
    target = train_data["user", "listing"].edge_label
    loss = weighted_mse_loss(pred, target, weight)
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test_for_rmse(data):
    model.eval()
    pred = model(data.x_dict, data.edge_index_dict, data["user", "listing"].edge_label_index)
    pred = pred.clamp(min=0, max=5)
    target = data["user", "listing"].edge_label.float()
    rmse = F.mse_loss(pred, target).sqrt()
    return float(rmse)


@torch.no_grad()
def test_for_hit_rate(data):
    model.eval()
    embeddings = model.inference(data.x_dict, data.edge_index_dict)
    user_embeddings = embeddings["user"]
    listing_embeddings = embeddings["listing"]
    return evaluate_nn(
        user_embeddings,
        listing_embeddings,
        test_reviews,
        test_listings,
        test_listings2dict,
        reverse_test_listings2dict,
        test_reviewers2dict,
        reverse_test_reviewers2dict,
    )


if __name__ == "__main__":
    directory = "../../../data/processed"
    separate_date = "2022-01"

    # Load data
    reviews = pd.read_parquet(f"{directory}/reviews_with_interactions.parquet")
    listings = pd.read_parquet(f"{directory}/listings_with_interactions.parquet")
    reviewers = pd.read_parquet(f"{directory}/reviewers_with_interactions.parquet")

    print("Full: ", len(reviews), len(listings), len(reviewers))
    # Prepare data and graph
    (
        train_reviews,
        train_listings,
        train_reviewers,
        test_reviews,
        test_listings,
        test_reviewers,
    ) = train_test_split(reviews, listings, reviewers, separate_date)
    test_listings2dict = get_entity2dict(test_listings, "listing_id")
    reverse_test_listings2dict = {k: v for v, k in test_listings2dict.items()}
    test_reviewers2dict = get_entity2dict(test_reviewers, "reviewer_id")
    reverse_test_reviewers2dict = {k: v for v, k in test_reviewers2dict.items()}

    data = build_heterograph(reviews, listings, True)
    train_data = build_heterograph(train_reviews, train_listings, True)
    test_data = build_heterograph(test_reviews, test_listings, True)

    # TODO: Add Neighbourloader
    # Modelling
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = train_data.to(device)

    # We have an unbalanced dataset with many labels for rating 3 and 4, and very
    # few for 0 and 1. Therefore we use a weighted MSE loss.
    if True:
        weight = torch.bincount(train_data["user", "listing"].edge_label)
        weight = weight.max() / weight
    else:
        weight = None

    model = Model(hidden_channels=32).to(device)

    with torch.no_grad():
        model.encoder(train_data.x_dict, train_data.edge_index_dict)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    epoch_list = []
    train_loss_list = []
    train_rmse_list = []
    test_rmse_list = []
    # TODO
    # Train and Evaluate
    is_export = False
    for epoch in range(1, 2):
        train_loss = train()
        train_rmse = test_for_rmse(train_data)
        test_rmse = test_for_rmse(test_data)

        if is_export:
            model_directory = "./models"
            torch.save(model.state_dict(), f"{model_directory}/{epoch}_model_state_dict.pt")
            torch.save(model, f"{model_directory}/{epoch}_model.pt")

        epoch_list.append(epoch)
        train_loss_list.append(train_loss)
        train_rmse_list.append(train_rmse)
        test_rmse_list.append(test_rmse)

        print(
            f"Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Train RMSE: {train_rmse:.4f}, "
            f"Test RMSE: {test_rmse:.4f}"
        )

    if is_export:

        training_history_dict = {
            "epoch": epoch_list,
            "train_loss": train_loss_list,
            "train_rmse": train_rmse_list,
            "test_rmse": test_rmse_list,
        }
        write_dict_into_json(training_history_dict, "./output/training_history.json")
