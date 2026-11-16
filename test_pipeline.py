import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import shutil
import numpy as np
import pandas as pd

from mlp.preprocessing import DatasetPreprocessor
from mlp.network import MultiLayerPerceptron
from split import split_dataset
from train import train
from predict import evaluate_predictions


def test_imputation_by_class():
    print("\n--- ТЕСТ 1: Проверка заполнения пропусков по классам ---")
    # Создаем мини-датасет с искусственными пропусками
    X_test = np.array([
        [10.0, 20.0],
        [np.nan, 30.0],
        [12.0, np.nan],
        [100.0, 200.0],
        [110.0, np.nan],
        [np.nan, 210.0]
    ])
    y_test = np.array([0, 0, 0, 1, 1, 1])  # Класс 0 и Класс 1

    prep = DatasetPreprocessor(normalization='none')
    prep.fit(X_test, y_test)
    X_imputed = prep.transform(X_test, y_test)

    # Класс 0 средние: feat0 = (10+12)/2 = 11, feat1 = (20+30)/2 = 25
    # Класс 1 средние: feat0 = (100+110)/2 = 105, feat1 = (200+210)/2 = 205
    assert not np.isnan(X_imputed).any(), "Остались NaN после импутации!"
    assert np.isclose(X_imputed[1, 0], 11.0), f"Ожидалось 11.0 для класс 0 feat 0, получено {X_imputed[1,0]}"
    assert np.isclose(X_imputed[2, 1], 25.0), f"Ожидалось 25.0 для класс 0 feat 1, получено {X_imputed[2,1]}"
    assert np.isclose(X_imputed[4, 1], 205.0), f"Ожидалось 205.0 для класс 1 feat 1, получено {X_imputed[4,1]}"
    assert np.isclose(X_imputed[5, 0], 105.0), f"Ожидалось 105.0 для класс 1 feat 0, получено {X_imputed[5,0]}"
    print("[УСПЕХ] Заполнение пропусков по среднему значению класса работает безупречно!")


def test_full_pipeline():
    print("\n--- ТЕСТ 2: Полный сквозной пайплайн (split -> train -> predict) ---")
    # 1. Split
    split_dataset('data.csv', train_ratio=0.8, seed=42, out_train='test_train.csv', out_val='test_val.csv')
    assert os.path.exists('test_train.csv')
    assert os.path.exists('test_val.csv')

    # 2. Train с Adam и ReLU
    model, history = train(
        dataset_path='test_train.csv',
        val_dataset_path='test_val.csv',
        layer_sizes=[24, 24],
        epochs=30,
        batch_size=16,
        learning_rate=0.01,
        optimizer_name='adam',
        activation_name='relu',
        model_out='test_model.npz',
        save_plot_path='plots/test_curves.png',
        show_plot=False
    )
    assert os.path.exists('test_model.npz')
    assert os.path.exists('plots/test_curves.png')
    val_acc = history['val_acc'][-1]
    print(f"Достигнутая точность на валидации за 30 эпох: {val_acc * 100:.2f}%")
    assert val_acc > 0.90, f"Точность слишком низкая: {val_acc}"

    # 3. Predict
    probas, preds = evaluate_predictions(model_path='test_model.npz', dataset_path='test_val.csv')
    assert len(preds) == len(probas)
    print("[УСПЕХ] Сквозной пайплайн отработал отлично!")

    # Удаляем временные тестовые файлы
    for f in ('test_train.csv', 'test_val.csv', 'test_model.npz', 'plots/test_curves.png'):
        if os.path.exists(f):
            os.remove(f)


if __name__ == '__main__':
    test_imputation_by_class()
    test_full_pipeline()
