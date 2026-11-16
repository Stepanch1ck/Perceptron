import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from mlp.preprocessing import DatasetPreprocessor


def run_eda(dataset_path: str = 'data.csv', plots_dir: str = 'plots'):
    os.makedirs(plots_dir, exist_ok=True)
    print("=" * 70)
    print("ИССЛЕДОВАТЕЛЬСКИЙ АНАЛИЗ ДАННЫХ")
    print("=" * 70)

    # 1. Загрузка
    preprocessor = DatasetPreprocessor()
    df = preprocessor.load_dataframe(dataset_path)
    print(f"\n1. Форма датасета: {df.shape[0]} строк (пациентов), {df.shape[1]} столбцов.")

    # 2. Проверка пропусков
    missing_count = df.isnull().sum().sum()
    print(f"2. Пропущенных значений в исходном датасете: {missing_count}")
    if missing_count > 0:
        print("   Пропуски по столбцам:")
        print(df.isnull().sum()[df.isnull().sum() > 0])
    else:
        print("   -> Данные полностью заполнены, средним значением по соответствующему классу.")

    # 3. Баланс классов
    target_col = 'diagnosis' if 'diagnosis' in df.columns else df.columns[1]
    class_counts = df[target_col].value_counts()
    class_pcts = df[target_col].value_counts(normalize=True) * 100
    print(f"\n3. Распределение целевой переменной ({target_col}):")
    for cls_name, count in class_counts.items():
        pct = class_pcts[cls_name]
        label_full = "Доброкачественная (Benign)" if cls_name == 'B' else "Злокачественная (Malignant)"
        print(f"   Класс '{cls_name}' ({label_full}): {count} образцов ({pct:.2f}%)")

    # График 1: Распределение классов
    plt.figure(figsize=(9, 4))
    plt.subplot(1, 2, 1)
    colors = ['#2ecc71', '#e74c3c']
    class_counts.plot(kind='bar', color=colors, edgecolor='black', alpha=0.85)
    plt.title('Количество образцов по классам', fontweight='bold')
    plt.xlabel('Диагноз (B / M)')
    plt.ylabel('Количество')
    plt.grid(axis='y', linestyle=':', alpha=0.6)

    plt.subplot(1, 2, 2)
    plt.pie(class_counts, labels=['Benign (B)', 'Malignant (M)'], autopct='%1.1f%%',
            colors=colors, startangle=140, explode=[0, 0.08],
            wedgeprops={'edgecolor': 'black', 'linewidth': 1})
    plt.title('Процентное соотношение', fontweight='bold')
    plt.tight_layout()
    plot_1_path = os.path.join(plots_dir, 'eda_class_distribution.png')
    plt.savefig(plot_1_path, dpi=300)
    plt.close()
    print(f"   [+] Сохранен график баланса классов: {plot_1_path}")

    # 4. Анализ ключевых признаков
    feature_cols = [c for c in df.columns if c not in ('id', 'diagnosis', 0, 1)]
    mean_cols = [c for c in feature_cols if 'mean' in str(c)]
    if not mean_cols:
        mean_cols = feature_cols[:10]

    print("\n4. Сравнение средних значений признаков для доброкачественных и злокачественных опухолей:")
    comparison = df.groupby(target_col)[mean_cols[:6]].mean().T
    comparison['Разница (%)'] = ((comparison['M'] - comparison['B']) / comparison['B']) * 100
    print(comparison.round(2))

    # График 2: Распределения ключевых признаков (KDE / Hist)
    key_features = mean_cols[:6]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for i, feat in enumerate(key_features):
        ax = axes[i]
        for cls_name, color, label in [('B', '#2ecc71', 'Benign (B)'), ('M', '#e74c3c', 'Malignant (M)')]:
            subset = df[df[target_col] == cls_name][feat].dropna()
            sns.histplot(subset, kde=True, color=color, label=label, ax=ax, alpha=0.4, stat='density')
        ax.set_title(f'Распределение {feat}', fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.legend()
    plt.suptitle('Сравнение распределений характеристик ядра клеток (Mean)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plot_2_path = os.path.join(plots_dir, 'eda_feature_distributions.png')
    plt.savefig(plot_2_path, dpi=300)
    plt.close()
    print(f"   [+] Сохранен график распределений признаков: {plot_2_path}")

    # График 3: Матрица корреляций первых 10 признаков
    plt.figure(figsize=(10, 8))
    corr_matrix = df[mean_cols].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1,
                linewidths=0.5, square=True)
    plt.title('Корреляционная матрица признаков группы Mean', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plot_3_path = os.path.join(plots_dir, 'eda_correlation_heatmap.png')
    plt.savefig(plot_3_path, dpi=300)
    plt.close()
    print(f"   [+] Сохранен график корреляционной матрицы: {plot_3_path}")


def main():
    dataset = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('-') else 'data.csv'
    plots_dir = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('-') else 'plots'
    run_eda(dataset_path=dataset, plots_dir=plots_dir)


if __name__ == '__main__':
    main()
