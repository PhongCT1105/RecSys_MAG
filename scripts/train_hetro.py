import argparse
import torch
import torch.nn as nn
from pathlib import Path
from torch_geometric.data import Data
import sys
from torch_geometric.loader import LinkNeighborLoader
import torch_geometric.transforms as T
import random
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(str(Path(__file__).parent))
from load_data import load_processed_data, get_hetrogeneous_data
from models.graph_sage_hetro import (
    GraphSAGE,
    train_epoch,
    train_epoch_bpr,
    evaluate_loader,
    evaluate_ranking_metrics,
)


def parse_args():
    parser = argparse.ArgumentParser(description="train graph neural network")
    parser.add_argument(
        "--model",
        type=str,
        default="sage",
        choices=["sage"],
        help="model architecture",
    )
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="weight decay")
    parser.add_argument("--hidden_dim", type=int, default=128, help="hidden dimension")
    parser.add_argument("--out_dim", type=int, default=64, help="output dimension")
    parser.add_argument("--num_layers", type=int, default=3, help="number of layers")
    parser.add_argument("--dropout", type=float, default=0.5, help="dropout rate")
    parser.add_argument(
        "--aggregator",
        type=str,
        default="mean",
        choices=["mean", "max", "lstm"],
        help="aggregator type",
    )
    parser.add_argument(
        "--include_journal",
        action="store_true",
        help="include journal index in features",
    )
    parser.add_argument("--log_interval", type=int, default=10, help="logging interval")
    parser.add_argument("--save_model", action="store_true", help="save trained model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./checkpoints/model.pt",
        help="path to save model",
    )
    parser.add_argument(
        "--random_features",
        action="store_true",
        help="use random features instead of specter2 embeddings",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument(
        "--use_bpr",
        action="store_true",
        help="use BPR loss instead of BCE loss",
    )

    return parser.parse_args()


def set_seed(seed):
    """Set seed for reproducibility across all random sources"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_loaders(train_data, val_data, test_data, batch_size=128, seed=42):
    """create link prediction loaders"""

    train_loader = LinkNeighborLoader(
        train_data,
        num_neighbors=[5, 2, 2],
        edge_label_index=("paper", "cites", "paper"),
        batch_size=batch_size,
        shuffle=True,
        neg_sampling_ratio=1.0,
        num_workers=0,
    )

    val_loader = LinkNeighborLoader(
        data=train_data,
        num_neighbors=[5, 2, 2],
        edge_label_index=(
            ("paper", "cites", "paper"),
            val_data["paper", "cites", "paper"].edge_label_index,
        ),
        edge_label=val_data["paper", "cites", "paper"].edge_label,
        batch_size=2 * batch_size,
        shuffle=False,
    )

    test_loader = LinkNeighborLoader(
        data=train_data,
        num_neighbors=[5, 2, 2],
        edge_label_index=(
            ("paper", "cites", "paper"),
            test_data["paper", "cites", "paper"].edge_label_index,
        ),
        edge_label=test_data["paper", "cites", "paper"].edge_label,
        batch_size=2 * batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("loading dataset...")
    train_data, val_data, test_data, data, mappings = get_hetrogeneous_data(
        seed=args.seed
    )
    print(train_data)

    train_loader, val_loader, test_loader = get_loaders(
        train_data, val_data, test_data, batch_size=256, seed=args.seed
    )

    if args.model == "sage":
        model = GraphSAGE(
            in_channels=data["paper"].num_features,
            hidden_channels=args.hidden_dim,
            out_channels=args.out_dim,
            metadata=data.metadata(),
            num_layers=args.num_layers,
            dropout=args.dropout,
            aggregator=args.aggregator,
            journal_embeddings=args.include_journal,
        )
    else:
        raise NotImplementedError("Other models loading not implemented")

    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"model: {args.model}")
    print(f"parameters: {num_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_val_auc = 0
    best_epoch = 0

    train_auc = evaluate_loader(model, train_loader, device)
    val_auc = evaluate_loader(model, val_loader, device)
    test_auc = evaluate_loader(model, test_loader, device)
    print(f"initial train auc: {train_auc:.4f}")
    print(f"initial val auc: {val_auc:.4f}")
    print(f"initial test auc: {test_auc:.4f}")
    print(f"\ntraining for {args.epochs} epochs...")

    for epoch in range(1, args.epochs + 1):

        if args.use_bpr:
            loss = train_epoch_bpr(model, train_loader, optimizer, device)
        else:
            loss = train_epoch(model, train_loader, optimizer, device)

        if epoch % args.log_interval == 0:

            train_auc = evaluate_loader(model, train_loader, device)
            val_auc = evaluate_loader(model, val_loader, device)

            print(
                f"epoch {epoch:03d} | loss: {loss:.4f} | train auc: {train_auc:.4f} | val auc: {val_auc:.4f}"
            )
            # print(
            #     f"  val p@1: {val_ranking['precision@1']:.4f} | "
            #     f"p@5: {val_ranking['precision@5']:.4f} | "
            #     f"p@10: {val_ranking['precision@10']:.4f} | "
            #     f"mrr: {val_ranking['mrr']:.4f}"
            # )

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch = epoch

                if args.save_model:
                    save_path = Path(args.model_path)
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "val_auc": val_auc,
                        },
                        save_path,
                    )
    if args.save_model:
        save_path = Path(args.model_path.replace(".pt", "_final.pt"))
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            save_path,
        )
    test_auc = evaluate_loader(model, test_loader, device)
    test_ranking = evaluate_ranking_metrics(model, test_loader, device, k_values=[1, 5])

    print(f"\ntraining complete!")
    print(f"best val auc: {best_val_auc:.4f} at epoch {best_epoch}")
    print(f"\ntest results:")
    print(f"  auc: {test_auc:.4f}")
    print(f"  precision@1: {test_ranking['precision@1']:.4f}")
    print(f"  precision@5: {test_ranking['precision@5']:.4f}")

    print(f"  mrr: {test_ranking['mrr']:.4f}")

    if args.save_model:
        print(f"model saved to {args.model_path}")


if __name__ == "__main__":
    main()
