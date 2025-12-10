from pathlib import Path
import json
import pandas as pd
from adapters import AutoAdapterModel
import torch
from transformers import AutoTokenizer
import torch_geometric.transforms as T
import numpy as np
from torch_geometric.data import Data, HeteroData
import pickle
from tqdm import tqdm
import random

JOURNAL_UNK = "<UNK_JOURNAL>"
AUTHOR_UNK = "<UNK_AUTHOR>"

ROOT = Path("..").resolve()
PATH_TRAIN = ROOT / "dataset" / "train.txt"
PATH_VAL = ROOT / "dataset" / "val.txt"
PATH_TEST = ROOT / "dataset" / "test.txt"


def load_split_whole_file(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as f:
        content = f.read()

    try:
        obj = json.loads(content)
    except json.JSONDecodeError:
        print(f"Failed to parse {path} as JSON.")
        exit(1)

    # we expect a *list of dicts*
    if not isinstance(obj, list):
        raise TypeError(f"Expected a list, got {type(obj)} from {path}")

    # one row per paper
    df = pd.json_normalize(obj)  # or pd.DataFrame(obj)
    return df


def parse_citations(cstr):
    if not isinstance(cstr, str):
        return []
    out = []
    for c in cstr.split(";"):
        c = c.strip()
        if c.isdigit():
            out.append(int(c))
    return out


def preprocess_papers(df: pd.DataFrame, min_journal_freq=5):
    df = df.copy()
    # remove duplicate publication_IDs
    df = df.drop_duplicates(subset=["publication_ID"])

    df = df[df["title"].notna() & df["abstract"].notna()]
    df = df[df["abstract"].str.len() > 20]
    df = df.reset_index(drop=True)

    df["citation_list"] = df["Citations"].apply(parse_citations)
    df["citation_count"] = df["citation_list"].str.len()

    df = df[df["citation_count"] > 0]  # atleast has cited one paper
    df["publication_ID"] = df["publication_ID"].astype(int)

    journal_counts = df["journal"].value_counts()
    df["cleaned_journal"] = df["journal"].copy()
    df.loc[
        df["cleaned_journal"].isin(
            journal_counts[journal_counts <= min_journal_freq].index
        ),
        "cleaned_journal",
    ] = JOURNAL_UNK

    return df


def make_mappings(df, min_author_papers=2):
    """create mappings for heterogeneous graph:
    1. paper -> cited papers (paper_cites_paper)
    2. paper -> journal (paper_belongs_to_journal)
    3. paper -> authors (paper_written_by_author)
    4. author -> author (author_coauthor_author)
    """

    # 1. paper to index mapping
    paper_ids = sorted(df["publication_ID"].unique())
    paper_id2idx = {pid: i for i, pid in enumerate(paper_ids)}
    idx2paper_id = {i: pid for pid, i in paper_id2idx.items()}

    # 2. journal to index mapping
    unique_journals = sorted(df["cleaned_journal"].unique())
    journal2idx = {j: i for i, j in enumerate(unique_journals)}
    idx2journal = {i: j for j, i in journal2idx.items()}

    # 3. count author paper frequency and replace rare authors with AUTHOR_UNK
    author_paper_count = {}
    for authors_list in df["authors"]:
        if isinstance(authors_list, list):
            for author in authors_list:
                if isinstance(author, dict) and "id" in author:
                    author_id = author["id"]
                    author_paper_count[author_id] = (
                        author_paper_count.get(author_id, 0) + 1
                    )

    # print distribution
    paper_counts = list(author_paper_count.values())
    print(f"\n=== author distribution ===")
    print(f"total unique authors: {len(author_paper_count)}")
    print(f"min papers per author: {min(paper_counts)}")
    print(f"max papers per author: {max(paper_counts)}")
    print(f"mean papers per author: {np.mean(paper_counts):.2f}")
    print(f"median papers per author: {np.median(paper_counts):.0f}")

    # count authors below threshold
    rare_authors = sum(1 for count in paper_counts if count < min_author_papers)
    print(
        f"authors with < {min_author_papers} papers: {rare_authors} ({rare_authors/len(author_paper_count)*100:.1f}%)"
    )

    # create cleaned author ids (replace rare authors with AUTHOR_UNK)
    cleaned_author_ids = {}
    for author_id, count in author_paper_count.items():
        if count >= min_author_papers:
            cleaned_author_ids[author_id] = author_id
        else:
            cleaned_author_ids[author_id] = AUTHOR_UNK

    # author to index mapping (including AUTHOR_UNK)
    unique_author_ids = sorted(set(cleaned_author_ids.values()))
    author_id2idx = {aid: i for i, aid in enumerate(unique_author_ids)}
    idx2author_id = {i: aid for aid, i in author_id2idx.items()}

    # 4. create edge indices
    # paper cites paper
    paper_cites_paper_src = []
    paper_cites_paper_dst = []

    for _, row in df.iterrows():
        src_pid = int(row["publication_ID"])
        src_idx = paper_id2idx[src_pid]

        for cited_pid in row["citation_list"]:
            if cited_pid in paper_id2idx:
                dst_idx = paper_id2idx[cited_pid]
                paper_cites_paper_src.append(src_idx)
                paper_cites_paper_dst.append(dst_idx)

    # paper belongs to journal
    paper_journal_src = []
    paper_journal_dst = []

    for _, row in df.iterrows():
        paper_idx = paper_id2idx[int(row["publication_ID"])]
        journal_idx = journal2idx[row["cleaned_journal"]]
        paper_journal_src.append(paper_idx)
        paper_journal_dst.append(journal_idx)

    # paper written by author (using cleaned author ids)
    paper_author_src = []
    paper_author_dst = []

    for _, row in df.iterrows():
        paper_idx = paper_id2idx[int(row["publication_ID"])]

        if isinstance(row["authors"], list):
            for author in row["authors"]:
                if isinstance(author, dict) and "id" in author:
                    author_id = author["id"]
                    # use cleaned author id (may be AUTHOR_UNK)
                    cleaned_author_id = cleaned_author_ids.get(author_id, AUTHOR_UNK)
                    if cleaned_author_id in author_id2idx:
                        author_idx = author_id2idx[cleaned_author_id]
                        paper_author_src.append(paper_idx)
                        paper_author_dst.append(author_idx)
    paper_author_src = []
    paper_author_dst = []

    for _, row in df.iterrows():
        paper_idx = paper_id2idx[int(row["publication_ID"])]

        if isinstance(row["authors"], list):
            for author in row["authors"]:
                if isinstance(author, dict) and "id" in author:
                    author_id = author["id"]
                    if author_id in author_id2idx:
                        author_idx = author_id2idx[author_id]
                        paper_author_src.append(paper_idx)
                        paper_author_dst.append(author_idx)

    # author coauthor author (undirected - store only one direction, using cleaned ids)
    author_coauthor_edges = set()

    for _, row in df.iterrows():
        if isinstance(row["authors"], list):
            author_indices = []
            for author in row["authors"]:
                if isinstance(author, dict) and "id" in author:
                    author_id = author["id"]
                    # use cleaned author id
                    cleaned_author_id = cleaned_author_ids.get(author_id, AUTHOR_UNK)
                    if cleaned_author_id in author_id2idx:
                        author_indices.append(author_id2idx[cleaned_author_id])

            # create edges between all pairs of coauthors (only one direction)
            for i in range(len(author_indices)):
                for j in range(i + 1, len(author_indices)):
                    # store as (min, max) to ensure single direction
                    a1, a2 = author_indices[i], author_indices[j]
                    author_coauthor_edges.add((min(a1, a2), max(a1, a2)))

    # convert set to lists
    author_coauthor_src = [edge[0] for edge in author_coauthor_edges]
    author_coauthor_dst = [edge[1] for edge in author_coauthor_edges]

    mappings = {
        "paper_id2idx": paper_id2idx,
        "idx2paper_id": idx2paper_id,
        "journal2idx": journal2idx,
        "idx2journal": idx2journal,
        "author_id2idx": author_id2idx,
        "idx2author_id": idx2author_id,
        "edges": {
            "paper_cites_paper": np.array(
                [paper_cites_paper_src, paper_cites_paper_dst]
            ),
            "paper_belongs_to_journal": np.array(
                [paper_journal_src, paper_journal_dst]
            ),
            "paper_written_by_author": np.array([paper_author_src, paper_author_dst]),
            "author_coauthor_author": np.array(
                [author_coauthor_src, author_coauthor_dst]
            ),
        },
        "num_papers": len(paper_ids),
        "num_journals": len(unique_journals),
        "num_authors": len(cleaned_author_ids),
    }

    return mappings


def embedd_papers(df, batch_size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
    # load proximity adapter
    model.load_adapter(
        "allenai/specter2", load_as="proximity", source="hf", set_active=True
    )
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")

    # concatenate title and abstract
    text_batch = (df["title"] + tokenizer.sep_token + df["abstract"]).tolist()

    all_embeddings = []

    # Process in batches
    for i in tqdm(range(0, len(text_batch), batch_size), desc="Embedding papers"):
        batch_texts = text_batch[i : i + batch_size]

        # preprocess the input
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=False,
            max_length=512,
        )

        # Move inputs to device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output = model(**inputs)
        # take the first token in the batch as the embedding
        batch_embeddings = output.last_hidden_state[:, 0, :].cpu()
        all_embeddings.append(batch_embeddings)

    # concatenate all embeddings
    embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"Final embeddings shape: {embeddings.shape}")
    return embeddings


def get_hetrogeneous_data(seed=42):
    mappings = load_processed_data()
    num_papers = mappings["num_papers"]
    journals = mappings["df"]["cleaned_journal"].tolist()
    authors = mappings["df"]["authors"].tolist()
    x_tensor = torch.tensor(
        mappings["paper_embeddings"], dtype=torch.float, requires_grad=False
    )
    data = HeteroData()
    data["paper"].x = x_tensor

    data["journal"].node_id = torch.tensor(
        [mappings["journal2idx"][j] for j in journals], dtype=torch.long
    )
    data["author"].node_id = torch.tensor(
        [
            mappings["author_id2idx"].get(
                author["id"], mappings["author_id2idx"][AUTHOR_UNK]
            )
            for authors_list in authors
            for author in authors_list
            if isinstance(author, dict) and "id" in author
        ],
        dtype=torch.long,
    )
    # ['paper_cites_paper', 'paper_belongs_to_journal',
    #  'paper_written_by_author', 'author_coauthor_author']
    data["paper", "cites", "paper"].edge_index = torch.tensor(
        mappings["edges"]["paper_cites_paper"]
    )
    data["paper", "belongs_to", "journal"].edge_index = torch.tensor(
        mappings["edges"]["paper_belongs_to_journal"]
    )
    data["paper", "written_by", "author"].edge_index = torch.tensor(
        mappings["edges"]["paper_written_by_author"]
    )
    data["author", "coauthor", "author"].edge_index = torch.tensor(
        mappings["edges"]["author_coauthor_author"]
    )
    data = T.ToUndirected()(data)

    # Remove edge_label attributes from reverse edges (if they exist)
    if hasattr(data["paper", "rev_cites", "paper"], "edge_label"):
        del data["paper", "rev_cites", "paper"].edge_label
    if hasattr(data["author", "rev_coauthor", "author"], "edge_label"):
        del data["author", "rev_coauthor", "author"].edge_label

    # Remove empty edge types (those without edge_index)
    edge_types_to_remove = []
    for edge_type in data.edge_types:
        if not hasattr(data[edge_type], "edge_index") or data[edge_type].num_edges == 0:
            edge_types_to_remove.append(edge_type)

    for edge_type in edge_types_to_remove:
        del data[edge_type]

    data.validate(raise_on_error=True)

    # Set seeds for reproducible data splitting
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_data, val_data, test_data = T.RandomLinkSplit(
        num_val=0.1,
        num_test=0.2,
        disjoint_train_ratio=0.3,  # 30% for supervision, 70% for message passing
        neg_sampling_ratio=2,
        add_negative_train_samples=False,
        edge_types=[
            ("paper", "cites", "paper"),
            ("paper", "belongs_to", "journal"),
            ("paper", "written_by", "author"),
            ("author", "coauthor", "author"),
        ],
        rev_edge_types=[
            ("journal", "rev_belongs_to", "paper"),
            ("author", "rev_written_by", "paper"),
            ("author", "rev_coauthor", "author"),
            ("paper", "rev_cites", "paper"),
        ],
    )(data)

    # Remove empty edge types from all data splits
    for dataset in [train_data, val_data, test_data]:
        edge_types_to_remove = []
        for edge_type in dataset.edge_types:
            if (
                not hasattr(dataset[edge_type], "edge_index")
                or dataset[edge_type].num_edges == 0
            ):
                edge_types_to_remove.append(edge_type)

        for edge_type in edge_types_to_remove:
            del dataset[edge_type]

    return train_data, val_data, test_data, data, mappings


def get_homogeneous_data(include_journal_idx=False, seed=42):
    """Load data and create train/val/test splits with reproducible seed"""
    mappings = load_processed_data()
    num_papers = mappings["num_papers"]
    pub_years = torch.tensor(
        mappings["df"]["pubDate"].str.split(" ").str[0].astype(int).values
    )
    x_tensor = torch.tensor(
        mappings["paper_embeddings"], dtype=torch.float, requires_grad=False
    )
    if include_journal_idx:
        journal_idx_tensor = torch.tensor(
            [
                mappings["journal2idx"].get(j, mappings["journal2idx"][JOURNAL_UNK])
                for j in mappings["df"]["cleaned_journal"]
            ],
            dtype=torch.long,
        )
        x_tensor = torch.cat(
            [x_tensor, journal_idx_tensor.unsqueeze(1).float()],
            dim=1,
        )
    data = Data(x=x_tensor, time=pub_years)
    data.edge_index = torch.tensor(
        mappings["edges"]["paper_cites_paper"], dtype=torch.long
    )
    data.num_nodes = num_papers
    data.validate(raise_on_error=True)

    # Set seeds for reproducible data splitting
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    transform = T.RandomLinkSplit(
        num_val=0.1,
        num_test=0.2,
        is_undirected=False,
        disjoint_train_ratio=0.3,  # 30% for supervision, 70% for message passing
        neg_sampling_ratio=2,
        add_negative_train_samples=False,
        # edge_types=("paper", "cites", "paper"),
        # rev_edge_types=("paper", "rev_cites", "paper"),
    )

    train_data, val_data, test_data = transform(data)
    return train_data, val_data, test_data, data, mappings


def load_processed_data():
    pickle_path = ROOT / "dataset" / "processed_data_rnd.pkl"
    if not pickle_path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {pickle_path}. Run load_data.py first or download from G-Drive."
        )
    with pickle_path.open("rb") as f:
        data = pickle.load(f)
    return data


if __name__ == "__main__":
    df_train = load_split_whole_file(PATH_TRAIN)
    df_val = load_split_whole_file(PATH_VAL)
    df_test = load_split_whole_file(PATH_TEST)

    df_all = pd.concat([df_train, df_val, df_test]).reset_index(drop=True)
    print(f"Total papers before preprocessing: {len(df_all)}")
    # print(df_all.columns)
    # Index(['publication_ID', 'Citations', 'pubDate', 'language', 'title',
    #    'journal', 'abstract', 'keywords', 'authors', 'venue', 'doi'],
    #   dtype='object')

    # preprocess the dataset
    df_all = preprocess_papers(df_all, min_journal_freq=2)
    print(f"Total papers after preprocessing: {len(df_all)}")

    # create mappings
    mappings = make_mappings(df_all)

    print(f"\n=== mapping statistics ===")
    print(f"num papers: {mappings['num_papers']}")
    print(f"num journals: {mappings['num_journals']}")
    print(f"num authors: {mappings['num_authors']}")

    print(f"\n=== edge statistics ===")
    for edge_type, edge_index in mappings["edges"].items():
        print(f"{edge_type}: {edge_index.shape}")

    # verify
    assert len(df_all) == 56416
    mappings["df"] = df_all
    embeddings = embedd_papers(df_all, batch_size=64)
    mappings["paper_embeddings"] = embeddings
    pickle_path = ROOT / "dataset" / "processed_data.pkl"
    with pickle_path.open("wb") as f:
        pickle.dump(mappings, f)
    print(f"\nSaved processed data to {pickle_path}")
