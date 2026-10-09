import numpy as np
from typing import List, Dict, Any, Optional

from .layers import DenseLayer
from .losses import get_loss, Loss
from .optimizers import get_optimizer, Optimizer
from .preprocessing import DatasetPreprocessor
from .metrics import accuracy_score


class MultiLayerPerceptron:
    def __init__(
        self,
        layer_sizes: Optional[List[int]] = None,
        activation: str = 'sigmoid',
        weights_initializer: str = 'heUniform',
        output_dim: int = 2,
        seed: Optional[int] = 42
    ):
        if layer_sizes is None:
            layer_sizes = [24, 24]

        self.hidden_sizes = list(layer_sizes)
        self.activation_name = activation
        self.weights_initializer = weights_initializer
        self.output_dim = output_dim
        self.seed = seed
        self.rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()

        self.layers: List[DenseLayer] = []
        self.preprocessor: Optional[DatasetPreprocessor] = None
        self.history: Dict[str, List[float]] = {
            'loss': [],
            'val_loss': [],
            'acc': [],
            'val_acc': []
        }
        self.is_built = False

    def build(self, input_dim: int):
        self.layers = []
        prev_dim = input_dim

        for hidden_dim in self.hidden_sizes:
            layer = DenseLayer(
                output_dim=hidden_dim,
                input_dim=prev_dim,
                activation=self.activation_name,
                weights_initializer=self.weights_initializer
            )
            layer.initialize(prev_dim, rng=self.rng)
            self.layers.append(layer)
            prev_dim = hidden_dim

        output_layer = DenseLayer(
            output_dim=self.output_dim,
            input_dim=prev_dim,
            activation='softmax',
            weights_initializer='xavierUniform'
        )
        output_layer.initialize(prev_dim, rng=self.rng)
        self.layers.append(output_layer)
        self.is_built = True

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray):
        dz = y_pred - y_true
        da = self.layers[-1].backward_from_dz(dz)
        for layer in reversed(self.layers[:-1]):
            da = layer.backward(da)

    def fit(
        self,
        X_train: np.ndarray,
        y_train_oh: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val_oh: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 8,
        learning_rate: float = 0.0314,
        optimizer_name: str = 'sgd',
        loss_name: str = 'binaryCrossEntropy',
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        if not self.is_built:
            self.build(input_dim=X_train.shape[1])

        loss_fn: Loss = get_loss(loss_name)
        optimizer: Optimizer = get_optimizer(optimizer_name, learning_rate=learning_rate)

        num_samples = X_train.shape[0]
        self.history = {'loss': [], 'val_loss': [], 'acc': [], 'val_acc': []}

        y_train_labels = np.argmax(y_train_oh, axis=1)
        if y_val_oh is not None:
            y_val_labels = np.argmax(y_val_oh, axis=1)

        for epoch in range(1, epochs + 1):
            indices = np.arange(num_samples)
            self.rng.shuffle(indices)
            X_shuffled = X_train[indices]
            y_shuffled = y_train_oh[indices]

            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                x_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                batch_pred = self.forward(x_batch)
                self.backward(y_true=y_batch, y_pred=batch_pred)

                for layer_idx, layer in enumerate(self.layers):
                    new_w, new_b = optimizer.update(
                        layer_idx=layer_idx,
                        w=layer.weights,
                        b=layer.biases,
                        dw=layer.dw,
                        db=layer.db
                    )
                    layer.weights = new_w
                    layer.biases = new_b

                optimizer.step()

            train_preds = self.forward(X_train)
            train_loss = loss_fn.forward(train_preds, y_train_oh)
            train_acc = accuracy_score(y_train_labels, np.argmax(train_preds, axis=1))

            self.history['loss'].append(train_loss)
            self.history['acc'].append(train_acc)

            val_str = ""
            if X_val is not None and y_val_oh is not None:
                val_preds = self.forward(X_val)
                val_loss = loss_fn.forward(val_preds, y_val_oh)
                val_acc = accuracy_score(y_val_labels, np.argmax(val_preds, axis=1))
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                val_str = f" - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}"

            if verbose:
                print(f"epoch {epoch:02d}/{epochs:02d} - loss: {train_loss:.4f} - acc: {train_acc:.4f}{val_str}")

        return self.history

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def predict(self, x: np.ndarray) -> np.ndarray:
        probas = self.predict_proba(x)
        return np.argmax(probas, axis=1)

    def save(self, filepath: str):
        save_dict = {
            'hidden_sizes': np.array(self.hidden_sizes),
            'activation_name': np.array(self.activation_name),
            'weights_initializer': np.array(self.weights_initializer),
            'output_dim': np.array(self.output_dim),
            'num_layers': np.array(len(self.layers))
        }

        for idx, layer in enumerate(self.layers):
            save_dict[f'layer_{idx}_w'] = layer.weights
            save_dict[f'layer_{idx}_b'] = layer.biases
            save_dict[f'layer_{idx}_act'] = np.array(layer.activation_name)

        if self.preprocessor is not None:
            prep_state = self.preprocessor.get_state()
            save_dict['has_preprocessor'] = np.array(True)
            save_dict['prep_normalization'] = np.array(prep_state['normalization'])
            if prep_state['mean'] is not None:
                save_dict['prep_mean'] = prep_state['mean']
                save_dict['prep_std'] = prep_state['std']
            if prep_state['min_val'] is not None:
                save_dict['prep_min_val'] = prep_state['min_val']
                save_dict['prep_max_val'] = prep_state['max_val']
            if prep_state['global_means'] is not None:
                save_dict['prep_global_means'] = prep_state['global_means']
            if prep_state['class_means_0'] is not None:
                save_dict['prep_class_means_0'] = prep_state['class_means_0']
            if prep_state['class_means_1'] is not None:
                save_dict['prep_class_means_1'] = prep_state['class_means_1']
        else:
            save_dict['has_preprocessor'] = np.array(False)

        np.savez_compressed(filepath, **save_dict)

    @classmethod
    def load(cls, filepath: str) -> 'MultiLayerPerceptron':
        data = np.load(filepath, allow_pickle=True)
        hidden_sizes = list(data['hidden_sizes'])
        activation_name = str(data['activation_name'])
        weights_initializer = str(data['weights_initializer'])
        output_dim = int(data['output_dim'])
        num_layers = int(data['num_layers'])

        model = cls(
            layer_sizes=hidden_sizes,
            activation=activation_name,
            weights_initializer=weights_initializer,
            output_dim=output_dim
        )

        model.layers = []
        for idx in range(num_layers):
            w = data[f'layer_{idx}_w']
            b = data[f'layer_{idx}_b']
            act = str(data[f'layer_{idx}_act'])
            input_dim = w.shape[0]
            output_layer_dim = w.shape[1]

            layer = DenseLayer(
                output_dim=output_layer_dim,
                input_dim=input_dim,
                activation=act,
                weights_initializer=weights_initializer
            )
            layer.weights = w
            layer.biases = b
            model.layers.append(layer)

        model.is_built = True

        if bool(data.get('has_preprocessor', False)):
            prep = DatasetPreprocessor(normalization=str(data['prep_normalization']))
            prep_state = {
                'normalization': str(data['prep_normalization']),
                'mean': data['prep_mean'] if 'prep_mean' in data else None,
                'std': data['prep_std'] if 'prep_std' in data else None,
                'min_val': data['prep_min_val'] if 'prep_min_val' in data else None,
                'max_val': data['prep_max_val'] if 'prep_max_val' in data else None,
                'global_means': data['prep_global_means'] if 'prep_global_means' in data else None,
                'class_means_0': data['prep_class_means_0'] if 'prep_class_means_0' in data else None,
                'class_means_1': data['prep_class_means_1'] if 'prep_class_means_1' in data else None,
                'is_fitted': True
            }
            prep.set_state(prep_state)
            model.preprocessor = prep

        return model
