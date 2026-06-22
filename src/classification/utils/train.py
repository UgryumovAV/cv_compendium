import argparse
import datetime
import os.path
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

from src.classification.datasets.dataloader import BalancedDataLoader
from src.classification.datasets.get_mnist import MNISTDataset
from src.classification.datasets.get_imagenet import IMAGENETDataset
from src.classification.models.lenet import LeNet
from src.classification.models.alexnet import AlexNet
from src.classification.utils.metrics import (calculate_metrics,
                                              plot_metrics_subplots, CalculateMetrics)
from src.utils.path_tools import create_log_file
from src.classification.utils.transformations import get_transforms


def train(
        device,
        epochs,
        train_loader,
        val_loader,
        model,
        criterion,
        optimizer,
        log_file,
        model_name: str
) -> None:
    validation_metrics = {
        "accuracy": list(),
        "precision": list(),
        "recall": list(),
        "f1_micro": list(),
        "f1_macro": list(),
        "roc_auc": list(),
    }

    metrics_calculator = CalculateMetrics(num_classes=10)

    # Training loop
    for epoch in range(epochs):
        start_time = datetime.datetime.now()
        model.train()
        running_loss = 0.0

        accuracy_train = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            output = model(data)
            loss = criterion(output, target)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            train_results = metrics_calculator.calculate_metrics(output.data, target)
            accuracy_train += train_results.accuracy

            # if (batch_idx + 1) % 100 == 0:
            #     print(f'Epoch: {epoch + 1}/{epochs} | '
            #           f'Batch: {batch_idx}/{len(train_loader)} | '
            #           f'Loss: {loss.item():.4f}')

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        accuracy = 0
        precision = 0
        recall = 0
        f1_micro = 0
        f1_macro = 0
        roc_auc = 0
        batch_numbers = 0

        with torch.no_grad():
            for data, target in val_loader:
                batch_numbers += 1
                data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
                outputs = model(data)
                loss = criterion(outputs, target)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

                # result_metrics = calculate_metrics(outputs.data, target)
                result_metrics = metrics_calculator.calculate_metrics(outputs.data, target)
                accuracy += result_metrics.accuracy
                precision += result_metrics.precision
                recall += result_metrics.recall
                f1_micro += result_metrics.f1_micro
                f1_macro += result_metrics.f1_macro
                roc_auc += result_metrics.roc_auc

        end_time = datetime.datetime.now()

        total_accuracy_train = accuracy_train / len(train_loader)
        total_accuracy = accuracy / batch_numbers
        total_precision = precision / batch_numbers
        total_recall = recall / batch_numbers
        total_f1_micro = f1_micro / batch_numbers
        total_f1_macro = f1_macro / batch_numbers
        total_roc_auc = roc_auc / batch_numbers
        result_string = (
            f'Epoch: {epoch + 1}/{epochs} | '
            f'Time: {end_time - start_time} | '
            f'Training Loss: {running_loss / len(train_loader):.3f} | '
            f'Validation Loss: {val_loss / len(val_loader):.3f} | '
            f'Training Accuracy: {total_accuracy_train:.3f}% | '
            f'Validation Accuracy: {total_accuracy:.3f}% | '
            f'Precision: {total_precision:.3f} | '
            f'Recall: {total_recall:.3f} | '
            f'F1-Micro: {total_f1_micro:.3f} | '
            f'F1-Macro: {total_f1_macro:.3f} | '
            f'ROC-AUC: {total_roc_auc:.3f}'
        )
        print(result_string)
        log_file.write(result_string + "\n")
        validation_metrics["accuracy"].append(total_accuracy)
        validation_metrics["precision"].append(total_precision)
        validation_metrics["recall"].append(total_recall)
        validation_metrics["f1_micro"].append(total_f1_micro)
        validation_metrics["f1_macro"].append(total_f1_macro)
        validation_metrics["roc_auc"].append(total_roc_auc)
        plot_metrics_subplots(validation_metrics, model_name)


def main(model_name: str) -> None:
    assert isinstance(model_name, str), ValueError('Model name must be a string')
    model_name = model_name.lower()
    assert model_name in ['lenet', 'alexnet'], ValueError('Model name must be in ["lenet", "alexnet"]')

    # Set log file
    log_file = create_log_file(model_name)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    batch_size = 8
    learning_rate = 0.01
    epochs = 10

    transformations_train, transformations_test = get_transforms(model_name)

    root_path = Path(__file__).resolve(strict=True).parent.parent.parent.parent
    # dataset_path = os.path.join(root_path, Path('data/datasets/mnist'))
    dataset_path = os.path.join(root_path, Path('data/datasets/imagenet'))

    # Load datasets
    # train_dataset = MNISTDataset(dataset_path, train=True, transform=transformations_train)
    # val_dataset = MNISTDataset(dataset_path, train=False, transform=transformations_test)
    train_dataset = IMAGENETDataset(dataset_path, split='train', transform=transformations_train)
    val_dataset = IMAGENETDataset(dataset_path, split='val', transform=transformations_test)

    # Create data loaders
    balanced_train_loader = BalancedDataLoader(
        train_dataset,
        batch_size=batch_size,
    )
    balanced_val_loader = BalancedDataLoader(
        val_dataset,
        batch_size=batch_size,
    )

    # Initialize model, loss function, and optimizer
    # model = LeNet(num_classes=10).to(device)
    model = AlexNet(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    train(
        device=device,
        epochs=epochs,
        train_loader=balanced_train_loader,
        val_loader=balanced_val_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        log_file=log_file,
        model_name=model_name
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--model_name', type=str, default='AlexNet', help='Model name')
    args = parser.parse_args()

    main(model_name=args.model_name)
