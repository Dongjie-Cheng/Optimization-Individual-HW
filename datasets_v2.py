# datasets_v2.py
from __future__ import annotations
from pathlib import Path
from typing import Tuple, List

import numpy as np

try:
    import scipy.sparse as sp
    from sklearn.datasets import load_svmlight_file
    _HAVE_SK = True
except Exception:
    sp = None
    load_svmlight_file = None
    _HAVE_SK = False


def _require_libsvm_stack():
    if not _HAVE_SK:
        raise ImportError(
            "datasets_v2 requires scikit-learn and scipy. "
            "Install with: pip install scikit-learn scipy"
        )


def _binarize_labels(y: np.ndarray) -> np.ndarray:
    """
    Map labels to {0,1}. For typical LIBSVM binary datasets, positive class
    is +1 (or >0), negative is -1 (or 0). We map >0 -> 1, else 0.
    """
    y = np.asarray(y).ravel()
    y_bin = (y > 0).astype(np.float64)
    return y_bin


def _pad_cols_to(X: "sp.spmatrix", n_features: int) -> "sp.csr_matrix":
    """
    If X has fewer than n_features columns, pad with zero columns on the right.
    If already has n_features, just return csr.
    """
    X = X.tocsr()
    cur = X.shape[1]
    if cur == n_features:
        return X
    if cur > n_features:
        raise ValueError(f"Matrix has {cur} features, which is > target {n_features}.")
    extra = n_features - cur
    pad = sp.csr_matrix((X.shape[0], extra), dtype=X.dtype)
    X_padded = sp.hstack([X, pad], format="csr")
    return X_padded


def load_libsvm_stack(
    train_path: str | Path,
    test_path: str | Path | None = None,
) -> Tuple["sp.csr_matrix", np.ndarray]:
    """
    Load a LIBSVM-format dataset (train + optional test) and stack them.

    Parameters
    ----------
    train_path : path to training file in LIBSVM format
    test_path  : optional path to test file; if provided, we vertically stack
                 train and test, making sure they have the same #features.

    Returns
    -------
    X : scipy.sparse.csr_matrix, shape (m, n)
    y : np.ndarray, shape (m,), labels in {0,1}
    """
    _require_libsvm_stack()

    train_path = Path(train_path)
    if not train_path.is_file():
        raise FileNotFoundError(f"LIBSVM train file not found: {train_path}")

    X_tr, y_tr = load_svmlight_file(str(train_path))
    y_tr = _binarize_labels(y_tr)

    X_list: List["sp.csr_matrix"] = [X_tr.tocsr()]
    y_list: List[np.ndarray] = [y_tr]

    if test_path is not None:
        test_path = Path(test_path)
        if not test_path.is_file():
            raise FileNotFoundError(f"LIBSVM test file not found: {test_path}")
        X_te, y_te = load_svmlight_file(str(test_path))
        y_te = _binarize_labels(y_te)

        # ---- 关键修复：统一特征维度 ----
        n_features = max(X_tr.shape[1], X_te.shape[1])
        X_tr_pad = _pad_cols_to(X_tr, n_features)
        X_te_pad = _pad_cols_to(X_te, n_features)

        X_list = [X_tr_pad, X_te_pad]
        y_list = [y_tr, y_te]

    X = sp.vstack(X_list, format="csr")
    y = np.concatenate(y_list, axis=0)
    return X, y


# -----------------------------
# Dataset-specific loaders
# -----------------------------
def load_a9a_binary(root: str | Path) -> Tuple["sp.csr_matrix", np.ndarray]:
    """
    Load the a9a dataset from LIBSVM format.

    Expected files (download from LIBSVM website and put under 'root'):
      - a9a    (train)
      - a9a.t  (test)

    Returns X (csr), y in {0,1}.
    """
    root = Path(root)
    train_path = root / "a9a"
    test_path = root / "a9a.t"
    return load_libsvm_stack(train_path, test_path)


def load_ijcnn1_binary(root: str | Path) -> Tuple["sp.csr_matrix", np.ndarray]:
    """
    Load the ijcnn1 dataset from LIBSVM format.

    Expected files under 'root':
      - ijcnn1    (train)
      - ijcnn1.t  (test)
    """
    root = Path(root)
    train_path = root / "ijcnn1"
    test_path = root / "ijcnn1.t"
    return load_libsvm_stack(train_path, test_path)


def load_realsim_binary(root: str | Path) -> Tuple["sp.csr_matrix", np.ndarray]:
    """
    Load the real-sim dataset from LIBSVM format.

    Expected file under 'root':
      - real-sim   (only one file; we treat all rows as a single pool, and
                    later split into train/val/test in your experiment code).
    """
    root = Path(root)
    path = root / "real-sim"
    return load_libsvm_stack(path, test_path=None)
