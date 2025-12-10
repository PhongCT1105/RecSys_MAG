import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv
from torch_geometric.data import HeteroData
from torch_geometric.utils import negative_sampling


JOURNAL_EMBEDDING_DIM = 16


class GraphSAGE(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        metadata,
        num_layers=2,
        dropout=0.5,
        aggregator="mean",
        journal_embeddings=False,
    ):
        super(GraphSAGE, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.journal_emb = nn.Embedding(2566, in_channels)
        self.author_emb = nn.Embedding(300059, in_channels)

        # Build heterogeneous conv layers manually
        self.convs = nn.ModuleList()

        # First layer: in_channels -> hidden_channels
        conv_dict = {}
        for edge_type in metadata[1]:
            src_type, _, dst_type = edge_type
            conv_dict[edge_type] = SAGEConv(
                (in_channels, in_channels), hidden_channels, aggr=aggregator
            )
        self.convs.append(HeteroConv(conv_dict, aggr="sum"))

        # Additional layers if needed
        if num_layers > 1:
            # Last layer: hidden_channels -> out_channels
            conv_dict = {}
            for edge_type in metadata[1]:
                src_type, _, dst_type = edge_type
                conv_dict[edge_type] = SAGEConv(
                    (hidden_channels, hidden_channels), out_channels, aggr=aggregator
                )
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))

        self.decoder = nn.Sequential(
            nn.Linear(2 * out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, 1),
        )

    def forward(self, data: HeteroData):
        x_dict = {
            "paper": data["paper"].x,
            "journal": self.journal_emb(data["journal"].node_id),
            "author": self.author_emb(data["author"].node_id),
        }

        # Apply first layer
        x_dict = self.convs[0](x_dict, data.edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = {
            key: F.dropout(x, p=self.dropout, training=self.training)
            for key, x in x_dict.items()
        }

        # Apply second layer if exists
        if self.num_layers > 1:
            x_dict = self.convs[1](x_dict, data.edge_index_dict)

        return x_dict

    def encode(self, data):
        """Get node embeddings (same as forward)."""
        return self.forward(data)

    def decode(self, z_dict, edge_label_index, edge_type=("paper", "cites", "paper")):
        """
        z_dict: Dictionary of node embeddings for each node type
        edge_label_index: Edge indices to predict [2, num_edges]
        edge_type: The edge type being predicted
        """
        # Get embeddings for the source and target node types
        src_type = edge_type[0]
        dst_type = edge_type[2]

        z_src = z_dict[src_type]
        z_dst = z_dict[dst_type]

        src = z_src[edge_label_index[0]]
        dst = z_dst[edge_label_index[1]]

        return self.decoder(torch.cat([src, dst], dim=-1)).view(-1)

    def decode_dot(
        self, z_dict, edge_label_index, edge_type=("paper", "cites", "paper")
    ):
        """
        Dot product decoder for BPR loss
        z_dict: Dictionary of node embeddings for each node type
        edge_label_index: Edge indices to predict [2, num_edges]
        edge_type: The edge type being predicted
        """
        src_type = edge_type[0]
        dst_type = edge_type[2]

        z_src = z_dict[src_type]
        z_dst = z_dict[dst_type]

        src = z_src[edge_label_index[0]]
        dst = z_dst[edge_label_index[1]]

        # Dot product between source and destination embeddings
        return (src * dst).sum(dim=-1)


def train_epoch(model, train_loader, optimizer, device, criterion=None):
    """train one epoch with minibatches for heterogeneous graphs"""
    model.train()

    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()

    total_loss = 0
    total_examples = 0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # encode - pass the heterogeneous batch
        z_dict = model.encode(batch)

        # decode link predictions for paper-cites-paper edges
        pred = model.decode(z_dict, batch["paper", "cites", "paper"].edge_label_index)

        # compute loss
        loss = criterion(pred, batch["paper", "cites", "paper"].edge_label.float())
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch["paper", "cites", "paper"].edge_label.size(0)
        total_examples += batch["paper", "cites", "paper"].edge_label.size(0)

    return total_loss / total_examples


def train_epoch_bpr(model, train_loader, optimizer, device):
    """Train one epoch with BPR loss for heterogeneous graphs"""
    model.train()

    total_loss = 0
    total_examples = 0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # Encode
        z_dict = model.encode(batch)

        # Separate positive and negative edges
        edge_label = batch["paper", "cites", "paper"].edge_label
        edge_label_index = batch["paper", "cites", "paper"].edge_label_index

        pos_mask = edge_label == 1
        neg_mask = edge_label == 0

        pos_edge_index = edge_label_index[:, pos_mask]
        neg_edge_index = edge_label_index[:, neg_mask]

        # Compute scores using dot product
        pos_scores = model.decode_dot(z_dict, pos_edge_index)
        neg_scores = model.decode_dot(z_dict, neg_edge_index)

        # BPR loss expects same number of pos and neg
        # If counts differ, sample or repeat to match
        if len(pos_scores) > len(neg_scores):
            # Repeat negatives to match positives
            repeats = (len(pos_scores) + len(neg_scores) - 1) // len(neg_scores)
            neg_scores = neg_scores.repeat(repeats)[: len(pos_scores)]
        elif len(neg_scores) > len(pos_scores):
            # Sample negatives to match positives
            indices = torch.randperm(len(neg_scores))[: len(pos_scores)]
            neg_scores = neg_scores[indices]

        # Compute BPR loss
        loss = bpr_loss(pos_scores, neg_scores)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(pos_scores)
        total_examples += len(pos_scores)

    return total_loss / total_examples


@torch.no_grad()
def evaluate_loader(model, loader, device):
    """evaluate using minibatch loader for heterogeneous graphs"""
    from sklearn.metrics import roc_auc_score

    model.eval()

    all_preds = []
    all_labels = []

    for batch in loader:
        batch = batch.to(device)
        # encode
        z_dict = model.encode(batch)

        # decode
        pred = model.decode(
            z_dict, batch["paper", "cites", "paper"].edge_label_index
        ).sigmoid()
        all_preds.append(pred.cpu())
        all_labels.append(batch["paper", "cites", "paper"].edge_label.cpu())

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
        z_dict = model.encode(batch)
        scores = model.decode(
            z_dict, batch["paper", "cites", "paper"].edge_label_index
        ).sigmoid()

        all_source_nodes.append(
            batch["paper", "cites", "paper"].edge_label_index[0].cpu()
        )
        all_target_nodes.append(
            batch["paper", "cites", "paper"].edge_label_index[1].cpu()
        )
        all_scores.append(scores.cpu())
        all_labels.append(batch["paper", "cites", "paper"].edge_label.cpu())

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
            if len(sorted_labels) >= k:
                precision_at_k[k].append(sorted_labels[:k].sum().item() / k)

        # mrr: find rank of first positive
        positive_indices = (sorted_labels == 1).nonzero(as_tuple=True)[0]
        if len(positive_indices) > 0:
            first_positive_rank = positive_indices[0].item() + 1
            reciprocal_ranks.append(1.0 / first_positive_rank)
        else:
            # No positive edges for this source node
            reciprocal_ranks.append(0.0)

    # compute averages
    metrics = {}
    for k in k_values:
        if precision_at_k[k]:
            metrics[f"precision@{k}"] = np.mean(precision_at_k[k])
        else:
            metrics[f"precision@{k}"] = np.nan

    metrics["mrr"] = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

    return metrics
