import argparse
import torch
import torch.nn as nn
from pathlib import Path
import sys
from torch_geometric.loader import LinkNeighborLoader
import torch_geometric.transforms as T

sys.path.append(str(Path(__file__).parent))
from load_data import load_dataset
from models.graph_sage import GraphSAGE, train_epoch, evaluate_loader


def parse_args():
    parser = argparse.ArgumentParser(description="train graph neural network")
    parser.add_argument(
        "--model", type=str, default="sage", choices=["sage"], help="model architecture"
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
    parser.add_argument("--seed", type=int, default=42, help="random seed")

    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_loaders(train_data, val_data, test_data):
    """create link prediction loaders with neighbor sampling"""
    num_neighbors = [10, 5]  # two layers, 10 neighbors each
    train_loader = LinkNeighborLoader(
        train_data,
        num_neighbors=num_neighbors,
        edge_label_index=train_data.edge_label_index,  # edges to predict
        batch_size=1000,
        shuffle=True,
        neg_sampling="binary",  # 1 negative per positive
        num_workers=0,
    )

    val_loader = LinkNeighborLoader(
        val_data,
        num_neighbors=num_neighbors,
        edge_label_index=val_data.edge_label_index,
        batch_size=500,
        shuffle=False,
        neg_sampling="binary",
    )

    test_loader = LinkNeighborLoader(
        test_data,
        num_neighbors=num_neighbors,
        edge_label_index=test_data.edge_label_index,
        batch_size=500,
        shuffle=False,
        neg_sampling="binary",
    )

    return train_loader, val_loader, test_loader


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    print("loading dataset...")
    (
        train_data,
        val_data,
        test_data,
        paper_id2idx,
        idx2paper_id,
        journal2idx,
        idx2journal,
    ) = load_dataset(include_journal_idx=args.include_journal)
    # data = data.to(device)
    print(train_data)
    print(val_data)
    train_loader, val_loader, test_loader = get_loaders(train_data, val_data, test_data)

    if args.model == "sage":
        model = GraphSAGE(
            in_channels=train_data.num_features,
            hidden_channels=args.hidden_dim,
            out_channels=args.out_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            aggregator=args.aggregator,
        )
    else:
        raise ValueError(f"unknown model: {args.model}")

    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"model: {args.model}")
    print(f"parameters: {num_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_val_auc = 0
    best_epoch = 0

    print(f"\ntraining for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, device)

        if epoch % args.log_interval == 0:
            train_auc = evaluate_loader(model, train_loader, device)
            val_auc = evaluate_loader(model, val_loader, device)

            print(
                f"epoch {epoch:03d} | loss: {loss:.4f} | train auc: {train_auc:.4f} | val auc: {val_auc:.4f}"
            )

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
                            "args": vars(args),
                        },
                        save_path,
                    )

    test_auc = evaluate_loader(model, test_loader, device)

    print(f"\ntraining complete!")
    print(f"best val auc: {best_val_auc:.4f} at epoch {best_epoch}")
    print(f"test auc: {test_auc:.4f}")

    if args.save_model:
        print(f"model saved to {args.model_path}")


if __name__ == "__main__":
    main()
