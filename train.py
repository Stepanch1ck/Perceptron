import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from mlp.network import MultiLayerPerceptron
from mlp.preprocessing import DatasetPreprocessor


def plot_learning_curves(history: dict, save_path: str = None, show: bool = False):
    epochs = range(1, len(history['loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(epochs, history['loss'], label='training loss', color='#2b5c8f')
    if 'val_loss' in history and len(history['val_loss']) > 0:
        ax1.plot(epochs, history['val_loss'], label='validation loss', color='#5c8f2b', linestyle='--')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend()

    ax2.plot(epochs, history['acc'], label='training acc', color='#e67e22')
    if 'val_acc' in history and len(history['val_acc']) > 0:
        ax2.plot(epochs, history['val_acc'], label='validation acc', color='#2980b9')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    if show:
        plt.show()
    plt.close()


def train(
    dataset_path: str = 'data_training.csv',
    val_dataset_path: str = 'data_validation.csv',
    layer_sizes: list = None,
    epochs: int = 70,
    batch_size: int = 8,
    learning_rate: float = 0.0314,
    loss_name: str = 'binaryCrossEntropy',
    optimizer_name: str = 'sgd',
    activation_name: str = 'sigmoid',
    weights_initializer: str = None,
    model_out: str = 'saved_model.npz',
    seed: int = 42,
    normalization: str = 'standard',
    save_plot_path: str = 'plots/learning_curves.png',
    show_plot: bool = False
):
    if layer_sizes is None:
        layer_sizes = [24, 24]

    if not os.path.exists(dataset_path):
        if os.path.exists('data.csv'):
            from split import split_dataset
            split_dataset('data.csv', train_ratio=0.8, seed=seed, out_train=dataset_path, out_val=val_dataset_path)
        else:
            raise FileNotFoundError(f"File not found: {dataset_path}")

    preprocessor = DatasetPreprocessor(normalization=normalization)
    df_train = preprocessor.load_dataframe(dataset_path)
    X_train_raw, y_train_labels, y_train_oh = preprocessor.parse_data(df_train)

    preprocessor.fit(X_train_raw, y_train_labels)
    X_train = preprocessor.transform(X_train_raw, y_train_labels)

    X_val = None
    y_val_oh = None
    if val_dataset_path and os.path.exists(val_dataset_path):
        df_val = preprocessor.load_dataframe(val_dataset_path)
        X_val_raw, y_val_labels, y_val_oh = preprocessor.parse_data(df_val)
        X_val = preprocessor.transform(X_val_raw, y_val_labels)

    print(f"x_train shape : {X_train.shape}")
    if X_val is not None:
        print(f"x_valid shape : {X_val.shape}")
    print()

    if weights_initializer is None:
        weights_initializer = 'heUniform' if activation_name.lower() in ('relu', 'leaky_relu') else 'xavierUniform'

    model = MultiLayerPerceptron(
        layer_sizes=layer_sizes,
        activation=activation_name,
        weights_initializer=weights_initializer,
        output_dim=2,
        seed=seed
    )
    model.build(input_dim=X_train.shape[1])
    model.preprocessor = preprocessor

    history = model.fit(
        X_train=X_train,
        y_train_oh=y_train_oh,
        X_val=X_val,
        y_val_oh=y_val_oh,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        optimizer_name=optimizer_name,
        loss_name=loss_name,
        verbose=True
    )

    print(f"\n> saving model '{model_out}' to disk...")
    model.save(model_out)

    plot_learning_curves(history, save_path=save_plot_path, show=show_plot)
    return model, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='data_training.csv')
    parser.add_argument('--val_dataset', default='data_validation.csv')
    parser.add_argument('--layer', nargs='+', type=int, default=[24, 24])
    parser.add_argument('--epochs', type=int, default=70)
    parser.add_argument('--loss', default='binaryCrossEntropy')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=0.0314)
    parser.add_argument('--optimizer', default='sgd')
    parser.add_argument('--activation', default='sigmoid')
    parser.add_argument('--model_out', default='saved_model.npz')
    args = parser.parse_args()

    train(
        dataset_path=args.dataset,
        val_dataset_path=args.val_dataset,
        layer_sizes=args.layer,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        loss_name=args.loss,
        optimizer_name=args.optimizer,
        activation_name=args.activation,
        model_out=args.model_out
    )


if __name__ == '__main__':
    main()
