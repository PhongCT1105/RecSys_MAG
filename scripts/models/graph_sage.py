import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import negative_sampling


class GraphSAGE(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers=3,
        dropout=0.5,
        aggregator="mean",
    ):
        super(GraphSAGE, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # First layer
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggregator))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(
                SAGEConv(hidden_channels, hidden_channels, aggr=aggregator)
            )
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.decoder = nn.Sequential(
            nn.Linear(2 * out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, 1),
        )

        # Output layer
        if num_layers > 1:
            self.convs.append(SAGEConv(hidden_channels, out_channels, aggr=aggregator))
        else:
            self.convs[0] = SAGEConv(in_channels, out_channels, aggr=aggregator)

    def forward(self, x, edge_index):
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Last layer without activation
        x = self.convs[-1](x, edge_index)

        return x

    def encode(self, x, edge_index):
        """Get node embeddings (same as forward)."""
        return self.forward(x, edge_index)

    def decode(self, z, edge_label_index):
        """
        z: Node embeddings [num_nodes, out_channels]
        edge_label_index: Edge indices to predict [2, num_edges]
        """
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]

        return self.decoder(torch.cat([src, dst], dim=-1)).view(-1)


def train_epoch(model, loader, optimizer, device, criterion=None):
    """train one epoch with minibatches"""
    model.train()

    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()

    total_loss = 0
    total_examples = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        # encode
        z = model.encode(batch.x, batch.edge_index)

        # decode link predictions
        pred = model.decode(z, batch.edge_label_index)

        # compute loss
        loss = criterion(pred, batch.edge_label.float())
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.edge_label.size(0)
        total_examples += batch.edge_label.size(0)

    return total_loss / total_examples


@torch.no_grad()
def evaluate_loader(model, loader, device):
    """evaluate using minibatch loader"""
    from sklearn.metrics import roc_auc_score

    model.eval()

    all_preds = []
    all_labels = []

    for batch in loader:
        batch = batch.to(device)
        # encode
        z = model.encode(batch.x, batch.edge_index)

        # decode
        pred = model.decode(z, batch.edge_label_index).sigmoid()

        all_preds.append(pred.cpu())
        all_labels.append(batch.edge_label.cpu())

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()

    return roc_auc_score(labels, preds)


@torch.no_grad()
def evaluate_ranking_metrics(model, loader, device, k_values=[1, 5]):
    """evaluate precision@k and mrr for link prediction"""
    import numpy as np

    model.eval()

    all_source_nodes = []
    all_target_nodes = []
    all_scores = []
    all_labels = []

    # collect all predictions
    for batch in loader:
        batch = batch.to(device)
        z = model.encode(batch.x, batch.edge_index)
        scores = model.decode(z, batch.edge_label_index).sigmoid()

        all_source_nodes.append(batch.edge_label_index[0].cpu())
        all_target_nodes.append(batch.edge_label_index[1].cpu())
        all_scores.append(scores.cpu())
        all_labels.append(batch.edge_label.cpu())

    source_nodes = torch.cat(all_source_nodes)
    target_nodes = torch.cat(all_target_nodes)
    scores = torch.cat(all_scores)
    labels = torch.cat(all_labels)

    # group by source node
    unique_sources = source_nodes.unique()

    precision_at_k = {k: [] for k in k_values}
    reciprocal_ranks = []

    for src in unique_sources:
        mask = source_nodes == src
        src_targets = target_nodes[mask]
        src_scores = scores[mask]
        src_labels = labels[mask]

        # sort by score descending
        sorted_indices = torch.argsort(src_scores, descending=True)
        sorted_labels = src_labels[sorted_indices]

        # precision@k
        for k in k_values:
            print("here", len(sorted_labels))
            if len(sorted_labels) >= k:
                precision_at_k[k].append(sorted_labels[:k].sum().item() / k)

        # mrr: find rank of first positive
        positive_indices = (sorted_labels == 1).nonzero(as_tuple=True)[0]
        if len(positive_indices) > 0:
            first_positive_rank = positive_indices[0].item() + 1
            reciprocal_ranks.append(1.0 / first_positive_rank)

    # compute averages
    metrics = {}
    for k in k_values:
        if precision_at_k[k]:
            metrics[f"precision@{k}"] = np.mean(precision_at_k[k])
        else:
            metrics[f"precision@{k}"] = 0.0

    metrics["mrr"] = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

    return metrics


def train_step(model, data, optimizer, criterion=None):
    """
    Single training step.

    Args:
        model: GraphSAGE model
        data: PyG Data object with train_pos_edge_index
        optimizer: Optimizer
        criterion: Loss function (default: BCEWithLogitsLoss)

    Returns:
        Loss value
    """
    model.train()
    optimizer.zero_grad()

    z = model.encode(data.x, data.train_pos_edge_index)

    # Positive edges
    pos_edge_index = data.train_pos_edge_index
    pos_pred = model.decode(z, pos_edge_index)

    # Negative sampling
    neg_edge_index = negative_sampling(
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=pos_edge_index.size(1),
    )
    neg_pred = model.decode(z, neg_edge_index)

    # Loss
    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()

    pos_loss = criterion(pos_pred, torch.ones_like(pos_pred))
    neg_loss = criterion(neg_pred, torch.zeros_like(neg_pred))
    loss = pos_loss + neg_loss

    loss.backward()
    optimizer.step()

    return loss.item()


if __name__ == "__main__":
    # Example usage
    from pathlib import Path
    import sys

    # Add scripts to path
    sys.path.append(str(Path(__file__).parent.parent / "scripts"))
    from load_data import load_dataset

    # Load data
    print("Loading dataset...")
    data = load_dataset(include_journal_idx=False)

    # Initialize model
    model = GraphSAGE(
        in_channels=data.x.size(1),
        hidden_channels=128,
        out_channels=64,
        num_layers=2,
        dropout=0.5,
    )

    print(f"Model: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    # Setup training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    data = data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # Training loop
    print("\nTraining...")
    for epoch in range(1, 101):
        loss = train_step(model, data, optimizer)

        if epoch % 10 == 0:
            train_auc = evaluate(model, data, data.train_pos_edge_index)
            val_auc = evaluate(model, data, data.val_pos_edge_index)
            print(
                f"Epoch {epoch:03d}, Loss: {loss:.4f}, "
                f"Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}"
            )

    # Final evaluation
    test_auc = evaluate(model, data, data.test_pos_edge_index)
    print(f"\nTest AUC: {test_auc:.4f}")
