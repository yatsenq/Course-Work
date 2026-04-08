"""
Experiment Utilities для курсової роботи
Функції для візуалізації, збереження метрик та завантаження результатів

Author: Student
Date: 2026-04-04
"""

import json
import os
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from tabulate import tabulate
import warnings
warnings.filterwarnings('ignore')

# Налаштування стилю графіків
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

# Для підтримки українських шрифтів
try:
    plt.rcParams['font.family'] = 'DejaVu Sans'
except:
    pass


def save_metrics(metrics_dict, path, model_name=None):
    """
    Зберігає метрики у JSON файл з timestamp
    
    Args:
        metrics_dict (dict): Словник з метриками
        path (str): Шлях для збереження
        model_name (str, optional): Назва моделі
    """
    output = {
        'timestamp': datetime.now().isoformat(),
        'metrics': metrics_dict
    }
    if model_name:
        output['model_name'] = model_name
    
    # Створити директорію якщо не існує
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Метрики збережено: {path}")


def load_all_results(results_dir):
    """
    Завантажує всі JSON файли з директорії
    
    Args:
        results_dir (str): Директорія з результатами
    
    Returns:
        dict: Словник {filename: data}
    """
    results = {}
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"⚠ Директорія не існує: {results_dir}")
        return results
    
    for json_file in results_path.glob('*.json'):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                results[json_file.stem] = json.load(f)
        except Exception as e:
            print(f"⚠ Помилка завантаження {json_file.name}: {e}")
    
    print(f"✓ Завантажено {len(results)} файлів результатів")
    return results


def save_confusion_matrix(y_true, y_pred, labels, path, title="Confusion Matrix"):
    """
    Створює та зберігає confusion matrix
    
    Args:
        y_true (array): Справжні мітки
        y_pred (array): Передбачені мітки
        labels (list): Назви класів
        path (str): Шлях для збереження
        title (str): Заголовок графіку
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'})
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Confusion matrix збережено: {path}")


def save_roc_curve(y_true, y_proba, path, title="ROC Curve"):
    """
    Створює та зберігає ROC криву
    
    Args:
        y_true (array): Справжні мітки (binary)
        y_proba (array): Ймовірності для позитивного класу
        path (str): Шлях для збереження
        title (str): Заголовок графіку
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ ROC крива збережена: {path}")


def save_learning_curves(train_losses, val_losses, path, title="Learning Curves"):
    """
    Створює та зберігає learning curves з анотацією фінальних значень
    
    Args:
        train_losses (list): Train loss по епохах
        val_losses (list): Validation loss по епохах
        path (str): Шлях для збереження
        title (str): Заголовок графіку
    """
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=6)
    plt.plot(epochs, val_losses, 'r-s', label='Val Loss', linewidth=2, markersize=6)
    
    # Анотації фінальних значень
    final_train = train_losses[-1]
    final_val = val_losses[-1]
    plt.annotate(f'{final_train:.4f}', 
                xy=(len(epochs), final_train), 
                xytext=(10, 0), textcoords='offset points',
                fontsize=10, color='blue', fontweight='bold')
    plt.annotate(f'{final_val:.4f}', 
                xy=(len(epochs), final_val), 
                xytext=(10, 0), textcoords='offset points',
                fontsize=10, color='red', fontweight='bold')
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Learning curves збережені: {path}")


def save_epoch_delta_chart(train_losses, val_losses, path, title="Epoch Delta Chart", threshold=0.05):
    """
    Створює bar plot різниці (val_loss - train_loss) по епохах
    Червоний колір якщо delta > threshold
    
    Args:
        train_losses (list): Train loss по епохах
        val_losses (list): Validation loss по епохах
        path (str): Шлях для збереження
        title (str): Заголовок графіку
        threshold (float): Поріг для визначення overfitting
    """
    epochs = range(1, len(train_losses) + 1)
    deltas = [val - train for train, val in zip(train_losses, val_losses)]
    colors = ['red' if d > threshold else 'green' for d in deltas]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(epochs, deltas, color=colors, alpha=0.7, edgecolor='black')
    
    # Горизонтальна лінія threshold
    plt.axhline(y=threshold, color='orange', linestyle='--', linewidth=2, 
                label=f'Threshold ({threshold})')
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    # Анотації значень
    for i, (epoch, delta, bar) in enumerate(zip(epochs, deltas, bars)):
        plt.text(bar.get_x() + bar.get_width()/2, delta, 
                f'{delta:.4f}', ha='center', va='bottom' if delta > 0 else 'top',
                fontsize=9, fontweight='bold')
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Delta (Val Loss - Train Loss)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=11)
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Epoch delta chart збережено: {path}")


def save_metric_per_epoch_comparison(metrics_dict, path, title="Metric Comparison", ylabel="Val Loss"):
    """
    Порівняння метрики кількох моделей по епохах на одному графіку
    
    Args:
        metrics_dict (dict): {model_name: [values_per_epoch]}
        path (str): Шлях для збереження
        title (str): Заголовок графіку
        ylabel (str): Підпис осі Y
    """
    plt.figure(figsize=(10, 6))
    
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']
    markers = ['o', 's', '^', 'D', 'v', 'p']
    
    for idx, (model_name, values) in enumerate(metrics_dict.items()):
        epochs = range(1, len(values) + 1)
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        
        plt.plot(epochs, values, marker=marker, color=color, linewidth=2, 
                markersize=7, label=model_name, alpha=0.8)
        
        # Анотація фінального значення
        plt.annotate(f'{values[-1]:.4f}', 
                    xy=(len(epochs), values[-1]), 
                    xytext=(10, 0), textcoords='offset points',
                    fontsize=9, color=color, fontweight='bold')
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Metric comparison збережено: {path}")


def save_bar_comparison(model_names, metrics_dict, path, title="Model Comparison"):
    """
    Створює grouped bar plot для порівняння метрик різних моделей
    
    Args:
        model_names (list): Назви моделей
        metrics_dict (dict): {metric_name: [values for each model]}
        path (str): Шлях для збереження
        title (str): Заголовок графіку
    """
    metric_names = list(metrics_dict.keys())
    num_metrics = len(metric_names)
    num_models = len(model_names)
    
    # Налаштування позицій барів
    x = np.arange(num_metrics)
    width = 0.8 / num_models
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.Set3(np.linspace(0, 1, num_models))
    
    for idx, model_name in enumerate(model_names):
        values = [metrics_dict[metric][idx] for metric in metric_names]
        offset = (idx - num_models/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=model_name, 
                     color=colors[idx], alpha=0.8, edgecolor='black')
        
        # Анотації значень
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}', ha='center', va='bottom', 
                   fontsize=8, fontweight='bold')
    
    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=10)
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3, axis='y')
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Bar comparison збережено: {path}")


def save_per_class_f1_bar(class_names, f1_scores, path, title="Per-Class F1 Scores"):
    """
    Створює bar plot F1-score для кожного класу
    
    Args:
        class_names (list): Назви класів
        f1_scores (list): F1 scores для кожного класу
        path (str): Шлях для збереження
        title (str): Заголовок графіку
    """
    plt.figure(figsize=(10, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(class_names)))
    bars = plt.bar(range(len(class_names)), f1_scores, color=colors, 
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Анотації значень
    for idx, (bar, score) in enumerate(zip(bars, f1_scores)):
        plt.text(bar.get_x() + bar.get_width()/2, score, 
                f'{score:.4f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold')
    
    plt.xlabel('Class', fontsize=12, fontweight='bold')
    plt.ylabel('F1 Score', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(range(len(class_names)), class_names, fontsize=10, rotation=45, ha='right')
    plt.ylim(0, 1.1)
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Per-class F1 bar збережено: {path}")


def print_metrics_table(metrics_dict, model_name="Model"):
    """
    Виводить таблицю метрик через tabulate
    
    Args:
        metrics_dict (dict): Словник з метриками
        model_name (str): Назва моделі
    """
    table_data = [[key, value] for key, value in metrics_dict.items()]
    
    print(f"\n{'='*50}")
    print(f"  {model_name} - Metrics Summary")
    print(f"{'='*50}")
    print(tabulate(table_data, headers=['Metric', 'Value'], 
                  tablefmt='grid', floatfmt='.4f'))
    print(f"{'='*50}\n")


# Додаткові допоміжні функції

def ensure_dir(path):
    """Створює директорію якщо не існує"""
    os.makedirs(path, exist_ok=True)


def save_top_features(feature_names, coefficients, n_top, path, title="Top Features"):
    """
    Зберігає топ-N найважливіших ознак для класифікації
    
    Args:
        feature_names (array): Назви ознак
        coefficients (array): Коефіцієнти моделі
        n_top (int): Кількість топ ознак
        path (str): Шлях для збереження
        title (str): Заголовок графіку
    """
    # Для binary classification
    if len(coefficients.shape) == 1:
        top_positive_idx = np.argsort(coefficients)[-n_top:]
        top_negative_idx = np.argsort(coefficients)[:n_top]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Positive features
        ax1.barh(range(n_top), coefficients[top_positive_idx], color='green', alpha=0.7)
        ax1.set_yticks(range(n_top))
        ax1.set_yticklabels([feature_names[i] for i in top_positive_idx], fontsize=9)
        ax1.set_xlabel('Coefficient', fontsize=11, fontweight='bold')
        ax1.set_title('Top Positive Features', fontsize=12, fontweight='bold')
        ax1.grid(alpha=0.3, axis='x')
        
        # Negative features
        ax2.barh(range(n_top), coefficients[top_negative_idx], color='red', alpha=0.7)
        ax2.set_yticks(range(n_top))
        ax2.set_yticklabels([feature_names[i] for i in top_negative_idx], fontsize=9)
        ax2.set_xlabel('Coefficient', fontsize=11, fontweight='bold')
        ax2.set_title('Top Negative Features', fontsize=12, fontweight='bold')
        ax2.grid(alpha=0.3, axis='x')
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Top features збережено: {path}")


if __name__ == "__main__":
    print("✓ Experiment utilities module loaded successfully")
    print("Available functions:")
    print("  - save_metrics()")
    print("  - load_all_results()")
    print("  - save_confusion_matrix()")
    print("  - save_roc_curve()")
    print("  - save_learning_curves()")
    print("  - save_epoch_delta_chart()")
    print("  - save_metric_per_epoch_comparison()")
    print("  - save_bar_comparison()")
    print("  - save_per_class_f1_bar()")
    print("  - print_metrics_table()")
    print("  - save_top_features()")
