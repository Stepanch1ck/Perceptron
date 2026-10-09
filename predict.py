import argparse
import os
import sys
import numpy as np
import pandas as pd

from mlp.network import MultiLayerPerceptron
from mlp.losses import BinaryCrossEntropy
from mlp.metrics import accuracy_score
from mlp.preprocessing import DatasetPreprocessor


def evaluate_predictions(
    model_path: str = 'saved_model.npz',
    dataset_path: str = 'data_validation.csv'
):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    model = MultiLayerPerceptron.load(model_path)
    preprocessor = model.preprocessor if model.preprocessor is not None else DatasetPreprocessor()

    df = preprocessor.load_dataframe(dataset_path)
    X_raw, y_labels, y_one_hot = preprocessor.parse_data(df)
    X_norm = preprocessor.transform(X_raw, y_labels=y_labels)

    probabilities = model.predict_proba(X_norm)
    predictions = np.argmax(probabilities, axis=1)

    if y_one_hot is not None:
        loss = BinaryCrossEntropy().forward(probabilities, y_one_hot)
        acc = accuracy_score(y_labels, predictions)
        correct = int(np.sum(y_labels == predictions))
        total = len(y_labels)
        print(f"loss: {loss:.4f} - accuracy: {acc:.4f} ({correct}/{total})")
    else:
        print(f"Predictions generated for {len(predictions)} samples.")

    return probabilities, predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='saved_model.npz')
    parser.add_argument('--dataset', default='data_validation.csv')
    args = parser.parse_args()

    evaluate_predictions(model_path=args.model, dataset_path=args.dataset)


if __name__ == '__main__':
    main()
