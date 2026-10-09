import os
import sys
import numpy as np
import pandas as pd


def split_dataset(
    dataset_path: str = 'data.csv',
    train_ratio: float = 0.8,
    seed: int = 42,
    out_train: str = 'data_training.csv',
    out_val: str = 'data_validation.csv'
):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"File not found: {dataset_path}")

    df = pd.read_csv(dataset_path, header=None)
    rng = np.random.RandomState(seed)

    if df.shape[1] >= 2:
        target_series = df.iloc[:, 1].astype(str).str.strip().str.upper()
        train_indices, val_indices = [], []

        for cls in target_series.unique():
            cls_indices = df[target_series == cls].index.to_numpy().copy()
            rng.shuffle(cls_indices)
            n_train_cls = int(np.round(len(cls_indices) * train_ratio))
            train_indices.extend(cls_indices[:n_train_cls])
            val_indices.extend(cls_indices[n_train_cls:])

        train_indices = np.array(train_indices)
        val_indices = np.array(val_indices)
        rng.shuffle(train_indices)
        rng.shuffle(val_indices)
    else:
        indices = np.arange(len(df))
        rng.shuffle(indices)
        n_train = int(np.round(len(df) * train_ratio))
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

    df_train = df.iloc[train_indices].reset_index(drop=True)
    df_val = df.iloc[val_indices].reset_index(drop=True)

    df_train.to_csv(out_train, header=False, index=False)
    df_val.to_csv(out_val, header=False, index=False)

    print(f"Data split: {len(df_train)} train -> {out_train}, {len(df_val)} valid -> {out_val}")


def main():
    dataset = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('-') else 'data.csv'
    split_dataset(dataset)


if __name__ == '__main__':
    main()
