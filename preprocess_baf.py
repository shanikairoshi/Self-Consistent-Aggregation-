from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, MinMaxScaler
from sklearn.decomposition import PCA


def _pack_xy_as_records(X: np.ndarray, y: np.ndarray) -> List[Dict[str, Any]]:
    return [
        {
            "sequence": X[i].astype(np.float32),
            "label": int(y[i]),
        }
        for i in range(len(y))
    ]


def _stratified_subsample(
    X: pd.DataFrame,
    y: np.ndarray,
    max_samples: Optional[int],
    seed: int,
):
    if max_samples is None or max_samples >= len(y):
        return X, y

    rng = np.random.default_rng(seed)
    selected = []

    for cls in np.unique(y):
        idx_cls = np.where(y == cls)[0]
        n_cls = max(1, int(max_samples * len(idx_cls) / len(y)))
        chosen = rng.choice(idx_cls, size=min(n_cls, len(idx_cls)), replace=False)
        selected.extend(chosen.tolist())

    selected = np.asarray(selected)
    rng.shuffle(selected)

    return X.iloc[selected].reset_index(drop=True), y[selected]


def load_and_prepare_dataset(
    *,
    csv_path: str,
    n_features: int = 4,
    global_seed: int = 42,
    target_col: str = "fraud_bool",
    test_size: float = 0.25,
    val_size: float = 0.15,
    max_samples: Optional[int] = 5000,
    scale_to_angles: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Load BAF: Bank Account Fraud dataset and convert it to DUQFL records.

    Returns:
        train_records, val_records, test_records
    """

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"BAF CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found. Available columns: {df.columns.tolist()}"
        )

    df = df.dropna().copy()

    y = df[target_col].astype(int).to_numpy()
    X_df = df.drop(columns=[target_col])

    # Optional subsampling because BAF can be large for QNN experiments.
    X_df, y = _stratified_subsample(
        X=X_df,
        y=y,
        max_samples=max_samples,
        seed=global_seed,
    )

    # Split before encoding/scaling to avoid leakage.
    X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
        X_df,
        y,
        test_size=test_size + val_size,
        random_state=global_seed,
        stratify=y,
    )

    relative_test_size = test_size / (test_size + val_size)

    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_temp_raw,
        y_temp,
        test_size=relative_test_size,
        random_state=global_seed,
        stratify=y_temp,
    )

    categorical_cols = X_train_raw.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_cols = [c for c in X_train_raw.columns if c not in categorical_cols]

    # Numeric part
    X_train_num = X_train_raw[numeric_cols].to_numpy(dtype=np.float32)
    X_val_num = X_val_raw[numeric_cols].to_numpy(dtype=np.float32)
    X_test_num = X_test_raw[numeric_cols].to_numpy(dtype=np.float32)

    # Categorical part
    if len(categorical_cols) > 0:
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )

        X_train_cat = encoder.fit_transform(X_train_raw[categorical_cols].astype(str))
        X_val_cat = encoder.transform(X_val_raw[categorical_cols].astype(str))
        X_test_cat = encoder.transform(X_test_raw[categorical_cols].astype(str))

        X_train_all = np.hstack([X_train_num, X_train_cat])
        X_val_all = np.hstack([X_val_num, X_val_cat])
        X_test_all = np.hstack([X_test_num, X_test_cat])
    else:
        X_train_all = X_train_num
        X_val_all = X_val_num
        X_test_all = X_test_num

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_all)
    X_val_scaled = scaler.transform(X_val_all)
    X_test_scaled = scaler.transform(X_test_all)

    if n_features > X_train_scaled.shape[1]:
        raise ValueError(
            f"n_features={n_features} is larger than available dimension={X_train_scaled.shape[1]}"
        )

    pca = PCA(n_components=n_features, random_state=global_seed)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    if scale_to_angles:
        angle_scaler = MinMaxScaler(feature_range=(0.0, np.pi))
        X_train_final = angle_scaler.fit_transform(X_train_pca)
        X_val_final = angle_scaler.transform(X_val_pca)
        X_test_final = angle_scaler.transform(X_test_pca)
    else:
        X_train_final = X_train_pca
        X_val_final = X_val_pca
        X_test_final = X_test_pca

    train_records = _pack_xy_as_records(X_train_final, y_train)
    val_records = _pack_xy_as_records(X_val_final, y_val)
    test_records = _pack_xy_as_records(X_test_final, y_test)

    print("=" * 70)
    print("BAF DATASET SUMMARY")
    print("=" * 70)
    print(f"CSV path: {csv_path}")
    print(f"Train samples: {len(train_records)}")
    print(f"Val samples:   {len(val_records)}")
    print(f"Test samples:  {len(test_records)}")
    print(f"QNN features/qubits: {n_features}")
    print(f"Train class distribution: {np.bincount(y_train)}")
    print(f"Val class distribution:   {np.bincount(y_val)}")
    print(f"Test class distribution:  {np.bincount(y_test)}")
    print("=" * 70)

    return train_records, val_records, test_records