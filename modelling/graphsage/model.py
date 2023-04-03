import torch
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import SAGEConv, to_hetero


class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
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
    neighbourhood_sampling_loader = LinkNeighborLoader(
        data,
        batch_size=batch_size,
        num_neighbors=num_neighbours,
        shuffle=True,
        neg_sampling_ratio=1.0,
        edge_label_index=(("user", "rates", "listing"), data["user", "listing"].edge_index),
    )
    return neighbourhood_sampling_loader


def load_model(path, hidden_channels, data):
    model = Model(hidden_channels=hidden_channels, data=data).to("cpu")
    model.load_state_dict(torch.load(path))
    return model
