from pathlib import Path
import json
import pandas as pd
from adapters import AutoAdapterModel
import torch
from transformers import AutoTokenizer
import numpy as np
from torch_geometric.data import Data

JOURNAL_UNK = "<UNK_JOURNAL>"
AUTHOR_UNK = "<UNK_AUTHOR>"

ROOT = Path("..").resolve()  # your notebook folder
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


def preprocess_papers(df: pd.DataFrame, min_journal_freq=5):
    df = df.copy()

    df = df[df["title"].notna() & df["abstract"].notna()]
    df = df[df["abstract"].str.len() > 20]

    def parse_citations(cstr):
        if not isinstance(cstr, str):
            return []
        out = []
        for c in cstr.split(";"):
            c = c.strip()
            if c.isdigit():
                out.append(int(c))
        return out

    df["citation_list"] = df["Citations"].apply(parse_citations)
    df["citation_count"] = df["citation_list"].str.len()

    df = df[df["citation_count"] > 0]
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


def embedd_papers(df, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
    # load proximity adapter
    model.load_adapter("allenai/specter2", source="hf", set_active=True)
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")

    # concatenate title and abstract
    text_batch = (df["title"] + tokenizer.sep_token + df["abstract"]).tolist()

    all_embeddings = []

    # Process in batches
    for i in range(0, len(text_batch), batch_size):
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

def get_journal_author_mappings(df):

def get_mappings(df):
    paper_ids = sorted(df["publication_ID"].unique().astype("int"))
    paper_id2idx = {pid: i for i, pid in enumerate(paper_ids)}
    idx2paper_id = {i: pid for pid, i in paper_id2idx.items()}

    unique_journals = sorted(df["cleaned_journal"].unique())
    journal2idx = {val: idx for idx, val in enumerate(unique_journals)}
    idx2journal = {idx: val for val, idx in journal2idx.items()}
    return paper_id2idx, idx2paper_id, journal2idx, idx2journal


def build_edges_for_split(df_split, paper_set, paper_id2idx):
    src_list, dst_list = [], []
    for _, row in df_split.iterrows():
        src_pid = int(row["publication_ID"])
        if src_pid not in paper_id2idx:
            print(f"Warning: source paper ID {src_pid} not in mapping, skipping")
            continue
        src_idx = paper_id2idx[src_pid]
        for cited_pid in row["citation_list"]:
            if cited_pid in paper_set:  # else cited paper not in dataset
                dst_idx = paper_id2idx[cited_pid]
                src_list.append(src_idx)
                dst_list.append(dst_idx)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    return edge_index


def check_transductive_inductive(df):
    """check if dataset is transductive or inductive by analyzing citation overlap"""
    df_tr = df[df["split"] == "train"]
    df_va = df[df["split"] == "val"]
    df_te = df[df["split"] == "test"]

    train_papers = set(df_tr["publication_ID"].astype("int"))
    val_papers = set(df_va["publication_ID"].astype("int"))
    test_papers = set(df_te["publication_ID"].astype("int"))

    def get_citations(df_split):
        citations = set()
        for _, row in df_split.iterrows():
            citations.update(row["citation_list"])
        return citations

    train_citations = get_citations(df_tr)
    val_citations = get_citations(df_va)
    test_citations = get_citations(df_te)

    # check train -> val/test
    train_cites_val = len(train_citations & val_papers)
    train_cites_test = len(train_citations & test_papers)

    # check val -> train/test
    val_cites_train = len(val_citations & train_papers)
    val_cites_test = len(val_citations & test_papers)

    # check test -> train/val
    test_cites_train = len(test_citations & train_papers)
    test_cites_val = len(test_citations & val_papers)

    print(f"\ntrain -> val citations: {train_cites_val}")
    print(f"train -> test citations: {train_cites_test}")
    print(f"val -> train citations: {val_cites_train}")
    print(f"val -> test citations: {val_cites_test}")
    print(f"test -> train citations: {test_cites_train}")
    print(f"test -> val citations: {test_cites_val}")

    is_transductive = (
        train_cites_val > 0
        or train_cites_test > 0
        or val_cites_train > 0
        or val_cites_test > 0
        or test_cites_train > 0
        or test_cites_val > 0
    )

    if is_transductive:
        print("TRANSDUCTIVE!!! splits share citation edges")
    else:
        print("INDUCTIVE!!! splits have no citation overlap")


def load_dataset(
    overwrite_preprocessed=False,
    include_journal_idx=True,
    preprocessed_path=ROOT / "preprocessed_df.pkl",
    embeddings_path=ROOT / "embeddings.npy",
):

    if preprocessed_path.exists() and not overwrite_preprocessed:
        print(f"Loading preprocessed dataframe from {preprocessed_path}")
        df = pd.read_pickle(preprocessed_path)
        print(df["split"].value_counts())
        embeddings = np.load(embeddings_path)
        print(f"Loaded {len(df)} preprocessed papers")
    else:
        print("No preprocessed dataframe found, processing from scratch...")
        df_train = load_split_whole_file(PATH_TRAIN)
        df_val = load_split_whole_file(PATH_VAL)
        df_test = load_split_whole_file(PATH_TEST)
        df_train["split"] = "train"
        df_val["split"] = "val"
        df_test["split"] = "test"

        df_raw = pd.concat([df_train, df_val, df_test], ignore_index=True)

        df = preprocess_papers(df_raw)
        df.to_pickle(preprocessed_path)
        embeddings = embedd_papers(df)
        np.save(embeddings_path, embeddings)
        print(f"Saved preprocessed df and embeddings")

    paper_ids = set(df["publication_ID"].astype(int))
    # track which papers have in-network citations
    papers_with_in_network_citations = set()
    for _, row in df.iterrows():
        source = int(row["publication_ID"])
        if type(row["Citations"]) != str:
            continue

        citations = str(row["Citations"]).split(";")
        for cite in citations:
            try:
                target = int(cite)
                # only add edge if target is in our dataset
                if target in paper_ids:
                    papers_with_in_network_citations.add(source)
                    papers_with_in_network_citations.add(target)
            except ValueError:
                continue
    # keep only papers with in-network citations
    # i.e., those that cite or are cited by another paper in the dataset
    # check_transductive_inductive(df)
    # get mappings for the full dataset

    paper_id2idx, idx2paper_id, journal2idx, idx2journal = get_mappings(df)
    # filter df to only include papers with in-network citations
    df = df[df["publication_ID"].astype(int).isin(papers_with_in_network_citations)]
    paper_set = set(df["publication_ID"].astype("int").tolist())
    valid_paper_idx = [paper_id2idx[pid] for pid in paper_set]
    embeddings = embeddings[valid_paper_idx, :].copy()  # filter embeddings to match df
    # reset index
    df = df.reset_index(drop=True)
    df_tr = df[df["split"] == "train"]
    df_va = df[df["split"] == "val"]
    df_te = df[df["split"] == "test"]
    # Sid is doing it this way because he already computed embeddings for all papers
    # and now is filtering to only those with in-network citations
    # so the indices still match up
    paper_id2idx, idx2paper_id, journal2idx, idx2journal = get_mappings(
        df
    )  # remap indices after filtering
    edge_index_train = build_edges_for_split(df_tr, paper_set, paper_id2idx)
    edge_index_val = build_edges_for_split(df_va, paper_set, paper_id2idx)
    edge_index_test = build_edges_for_split(df_te, paper_set, paper_id2idx)
    print(
        "Edge index shapes:",
        edge_index_train.shape,
        edge_index_val.shape,
        edge_index_test.shape,
    )
    if include_journal_idx:
        journal_indices = [journal2idx[j] for j in df["cleaned_journal"].tolist()]
        x = np.hstack([np.array(embeddings), np.array(journal_indices).reshape(-1, 1)])
    else:
        x = np.array(embeddings)
    x_tensor = torch.tensor(x, dtype=torch.float32)
    # x_tensor = torch.zeros_like(x_tensor)  # Sid is doing random features for now
    print(x_tensor)
    data = Data(x=x_tensor, edge_index=edge_index_train)
    data.val_pos_edge_index = edge_index_val
    data.test_pos_edge_index = edge_index_test
    data.train_pos_edge_index = edge_index_train
    # train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    # train_mask[[paper_id2idx[pid] for pid in df_tr["publication_ID"].astype(int)]] = (
    #     True
    # )
    # val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    # val_mask[[paper_id2idx[pid] for pid in df_va["publication_ID"].astype(int)]] = True
    # test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    # test_mask[[paper_id2idx[pid] for pid in df_te["publication_ID"].astype(int)]] = True
    # data.train_mask = train_mask
    # data.val_mask = val_mask
    # data.test_mask = test_mask

    data.validate(True)

    return data, paper_id2idx, idx2paper_id, journal2idx, idx2journal


if __name__ == "__main__":
    df_train = load_split_whole_file(PATH_TRAIN)
    df_val = load_split_whole_file(PATH_VAL)
    df_test = load_split_whole_file(PATH_TEST)
    df_train["split"] = "train"
    df_val["split"] = "val"
    df_test["split"] = "test"

    df_raw = pd.concat([df_train, df_val, df_test], ignore_index=True)

    df = preprocess_papers(df_raw)
    embeddings = embedd_papers(df, batch_size=256)
    df.to_pickle(ROOT / "preprocessed_df.pkl")
    np.save(ROOT / "embeddings.npy", embeddings)
