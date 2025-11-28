# data.py
from __future__ import annotations
import numpy as np

try:
    import scipy.sparse as sp
    _HAVE_SP = True
except Exception:
    sp = None
    _HAVE_SP = False

from sklearn.datasets import fetch_20newsgroups, fetch_rcv1
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


# -----------------------------
# 20 Newsgroups (binary) loader
# -----------------------------
def load_20ng_binary(
    vocab_size: int = 50_000,
    min_df: int = 5,
    positive_cats: tuple[str, ...] = (
        "comp.graphics",
        "comp.os.ms-windows.misc",
        "comp.sys.ibm.pc.hardware",
        "comp.sys.mac.hardware",
        "comp.windows.x",
    ),
    remove=("headers", "footers", "quotes"),
    stop_words: str | None = "english",
    tfidf: bool = True,
    random_state: int = 0,
):
    """
    Build a binary task on 20NG: positive = union(positive_cats), negative = the rest.
    Returns:
      X (csr_matrix), y (np.ndarray of shape (m,), values in {0,1}), target_names (list)
    """
    raw = fetch_20newsgroups(subset="all", remove=remove)
    texts = raw.data
    targets = raw.target
    names = list(raw.target_names)

    pos_idx = {names.index(cat) for cat in positive_cats if cat in names}
    y = np.array([1 if t in pos_idx else 0 for t in targets], dtype=np.float64)

    if tfidf:
        vec = TfidfVectorizer(max_features=vocab_size, min_df=min_df, stop_words=stop_words)
    else:
        vec = TfidfVectorizer(max_features=vocab_size, min_df=min_df, stop_words=stop_words, use_idf=False, norm=None)

    X = vec.fit_transform(texts)  # CSR
    return X, y, names


# -----------------------------
# RCV1 (binary) loader
# -----------------------------
def load_rcv1_binary(
    label: str = "CCAT",  # common top-level category
    subset: str = "all",  # "train", "test", or "all"
):
    """
    Load RCV1 and build y = 1{label present} from multilabel targets.
    Returns:
      X (csr_matrix), y (np.ndarray of shape (m,), values in {0,1}), target_names (list[str])
    Note: RCV1 is large; ensure you have RAM for subset='all'.
    """
    rcv = fetch_rcv1(subset=subset, download_if_missing=True)
    X = rcv.data  # CSR
    label_names = list(rcv.target_names)
    if label not in label_names:
        raise ValueError(f"Label '{label}' not in RCV1 target_names.")

    j = label_names.index(label)
    # rcv.target is CSR (m x n_labels); take column j
    y = np.array(rcv.target.getcol(j).toarray()).ravel().astype(np.float64)
    return X, y, label_names


# -----------------------------
# Sparse-preserving split
# -----------------------------
def train_val_test_split_sparse(
    X, y: np.ndarray,
    train_size: float = 0.7,
    val_size: float = 0.1,
    test_size: float = 0.2,
    seed: int = 0,
    stratify: bool = True,
):
    """
    Stratified (optional) split that preserves sparse CSR matrices.
    Returns: (X_tr, y_tr), (X_val, y_val), (X_te, y_te)
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-9, "splits must sum to 1"

    strat = y if stratify else None
    X_tmp, X_te, y_tmp, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=strat
    )
    # recompute val fraction relative to remaining
    val_rel = val_size / (train_size + val_size)
    strat_tmp = y_tmp if stratify else None
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_rel, random_state=seed, stratify=strat_tmp
    )
    return (X_tr, y_tr), (X_val, y_val), (X_te, y_te)


# -----------------------------
# Utility: ensure CSR
# -----------------------------
def ensure_csr(X):
    if _HAVE_SP and sp.issparse(X):
        return X.tocsr()
    return X
