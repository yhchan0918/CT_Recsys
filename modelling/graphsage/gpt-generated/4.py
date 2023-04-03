class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphSAGE, self).__init__()

        self.lin1 = torch.nn.Linear(in_channels, out_channels)
        self.lin2 = torch.nn.Linear(in_channels + out_channels, out_channels)

    def forward(self, x, edge_index, edge_weight):
        # Start message passing
        row, col = edge_index

        # Compute weighted message sum for each node
        weighted_messages = x[col] * edge_weight.unsqueeze(-1)
        aggr_messages = torch_scatter.scatter_add(
            weighted_messages, row, dim=0, dim_size=x.shape[0]
        )

        # Concatenate original node features with aggregated messages
        x = torch.cat([x, aggr_messages], dim=1)

        # Apply linear transformation and activation function
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(torch.cat([x, aggr_messages], dim=1))

        # Normalize node embeddings
        x = F.normalize(x, p=2, dim=1)

        return x


def test(model, test_loader, margin, device):
    model.eval()
    loss_fn = TripletMarginLoss(margin=margin)
    with torch.no_grad():
        test_loss = 0
        num_correct = 0
        num_total = 0
        for data in test_loader:
            data = data.to(device)
            out = model(data.x_dict, data.edge_index_dict)
            u_emb, v_emb, l_emb = out[data.u_idx], out[data.v_idx], out[data.l_idx]
            loss = loss_fn(u_emb, v_emb, l_emb)
            test_loss += loss.item() * len(data.u_idx)
            pred = torch.where(
                torch.sum(torch.abs(u_emb - v_emb), dim=1)
                < torch.sum(torch.abs(u_emb - l_emb), dim=1),
                1,
                0,
            )
            num_correct += torch.sum(pred == data.y).item()
            num_total += len(data.y)
        test_loss /= len(test_loader.dataset)
        accuracy = num_correct / num_total
    print("Test Loss: {:.4f}, Accuracy: {:.4f}".format(test_loss, accuracy))
