import os.path
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchmetrics import (AUROC, Accuracy, AveragePrecision, F1Score,
                          Precision, Recall, ConfusionMatrix, ROC)
from src.utils.path_tools import get_root


class Metrics:
    accuracy = None
    precision = None
    recall = None
    f1_micro = None
    f1_macro = None
    roc_auc = None


class CalculateMetrics:
    def __init__(
            self,
            num_classes: int,
    ):
        self.accuracy_function = Accuracy(task='multiclass', num_classes=num_classes)
        self.precision_function = Precision(task='multiclass', num_classes=num_classes)
        self.recall_function = Recall(task='multiclass', num_classes=num_classes)
        self.f1_micro_function = F1Score(task='multiclass', num_classes=num_classes, average='micro')
        self.f1_macro_function = F1Score(task='multiclass', num_classes=num_classes, average='macro')
        self.roc_auc_function = AUROC(task='multiclass', num_classes=num_classes)
        self.confusion_matrix_function = ConfusionMatrix(task='multiclass', num_classes=num_classes)

    def calculate_metrics(self, predicted: torch.Tensor, target: torch.Tensor) -> Metrics:
        predicted = predicted.to("cpu", non_blocking=True)
        target = target.to("cpu", non_blocking=True)
        result = Metrics()
        result.accuracy = self.accuracy_function(predicted, target)
        result.precision = self.precision_function(predicted, target)
        result.recall = self.recall_function(predicted, target)
        result.f1_micro = self.f1_micro_function(predicted, target)
        result.f1_macro = self.f1_macro_function(predicted, target)
        result.roc_auc = self.roc_auc_function(predicted, target)
        return result


def calculate_metrics(predicted: torch.Tensor, target: torch.Tensor) -> Metrics:
    predicted = predicted.to("cpu", non_blocking=True)
    target = target.to("cpu", non_blocking=True)

    result = Metrics()

    accuracy = Accuracy(task='multiclass', num_classes=10)
    precision = Precision(task='multiclass', num_classes=10)
    recall = Recall(task='multiclass', num_classes=10)
    f1_micro = F1Score(task='multiclass', num_classes=10, average='micro')
    f1_macro = F1Score(task='multiclass', num_classes=10, average='macro')
    roc_auc = AUROC(task='multiclass', num_classes=10)

    result.accuracy = accuracy(predicted, target)
    result.precision = precision(predicted, target)
    result.recall = recall(predicted, target)
    result.f1_micro = f1_micro(predicted, target)
    result.f1_macro = f1_macro(predicted, target)
    result.roc_auc = roc_auc(predicted, target)
    return result


# Macro-P = (P₁ + P₂ + ... + Pₖ) / K
# Macro-R = (R₁ + R₂ + ... + Rₖ) / K
# Macro-F1 = (F1₁ + F2₂ + ... + F1ₖ) / K or Macro-F1 = 2 × (Macro-P × Macro-R) / (Macro-P + Macro-R)

# Micro calculate global TP, FP, FN
# Global_TP = TP₁ + TP₂ + ... + TPₖ
# Global_FP = FP₁ + FP₂ + ... + FPₖ
# Global_FN = FN₁ + FN₂ + ... + FNₖ
# Micro-P = Global_TP / (Global_TP + Global_FP)
# Micro-R = Global_TP / (Global_TP + Global_FN)
# Micro-F1 = 2 × (Micro-P × Micro-R) / (Micro-P + Micro-R)

# Weighted-P = Σ(wᵢ × Pᵢ)
# where wᵢ = nᵢ / N (part of i class in data)

# ROC-AUC = the area under the curve ROC
# The curve ROC: TPR (Recall) vs FPR
# TPR = TP / (TP + FN)
# FPR = FP / (FP + TN)

# Multiclass ROC-AUC
# One-vs-Rest (OvR) / One-vs-All
# OvR ROC-AUC = (AUC₁ + AUC₂ + ... + AUCₖ) / K
# One-vs-One (OvO)
# OvO ROC-AUC = Σ AUCᵢⱼ / (K × (K-1) / 2)

# Recommendations for choosing the type of ROC-AUC:
#
# 1. BINARY ROC-AUC:
#  - Binary classification
#  - One-vs-Rest for multi-class
#
# 2. OVR (One-vs-Rest):
#  - Standard for multi-class classification
#  - Macro: all classes are equally important
#  - Weighted: account for unbalanced data
#
# 3. OVO (One-vs-One):
#  - When there are few classes (3-5)
#  - When pairwise comparisons are important
#  - Computationally expensive for many classes
#
# 4. MICRO:
#  - When overall performance is important
#  - For unbalanced data
#  - In multilabel classification
#
# 5. MULTILABEL:
#  - Macro: each label is equally important
#  - Micro: overall performance
#  - Samples: evaluation for each object


def plot_metrics_subplots(
        validation_metrics: dict,
        model_name
) -> None:
    root_path = get_root()
    log_file_dir_path = os.path.join(root_path, Path("data/logs"), Path(model_name), Path("images"))
    if os.path.exists(log_file_dir_path):
        shutil.rmtree(log_file_dir_path)
    Path(log_file_dir_path).mkdir(parents=True, exist_ok=True)
    save_path = os.path.join(log_file_dir_path, Path("metrics.png"))

    metrics_list = list(validation_metrics.keys())
    n_metrics = len(metrics_list)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    epochs = range(1, len(validation_metrics["accuracy"]) + 1)

    for i, metric_name in enumerate(metrics_list):
        if i < len(axes):
            values = validation_metrics[metric_name]
            if values:
                axes[i].plot(epochs, values, 'b-o', markersize=4, linewidth=2)
                axes[i].set_title(metric_name.upper())
                axes[i].set_xlabel('Epoch')
                axes[i].set_ylabel('Score')
                axes[i].grid(True, alpha=0.3)
                axes[i].set_ylim(0, 1)
                axes[i].xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    for i in range(n_metrics, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches='tight'
    )
