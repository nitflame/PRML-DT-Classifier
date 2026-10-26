import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def compute_diagnostic_metrics(y_true, y_pred, model_name: str) -> dict:
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    return {
        "Model": model_name,
        "Accuracy": acc,
        "Precision (Weighted)": prec,
        "Recall (Weighted)": rec,
        "F1-Score (Weighted)": f1
    }

def generate_confusion_matrix_plot(y_true, y_pred, target_labels=None):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=target_labels if target_labels is not None else True,
                yticklabels=target_labels if target_labels is not None else True, ax=ax)
    ax.set_xlabel('Predicted Model Target')
    ax.set_ylabel('True Reference Label')
    plt.tight_layout()
    return fig
