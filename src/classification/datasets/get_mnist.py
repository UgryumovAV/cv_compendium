import argparse
import os.path
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class MNISTDataset(Dataset):
    """Кастомный Dataset для MNIST"""

    def __init__(
            self,
            dataset_path: str = './data',
            train: bool = True,
            download: bool = False,
            transform: transforms.Compose | None = None
    ):
        self.mnist = torchvision.datasets.MNIST(
            root=dataset_path,
            train=train,
            download=download,
            transform=None
        )
        self.transform = transform

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        image, label = self.mnist[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def show_one_per_class(dataset, dataset_path):
    """
    Находит по одному примеру каждого класса (0-9)
    и отображает их в одном окне с использованием OpenCV
    """
    # Собираем по одному примеру каждого класса
    class_samples = {}
    for idx in range(len(dataset)):
        image, label = dataset[idx]
        label = label.item() if torch.is_tensor(label) else label

        if label not in class_samples:
            class_samples[label] = image
            if len(class_samples) == 10:  # Все классы собраны
                break

    # Проверяем, что все классы найдены
    if len(class_samples) != 10:
        print(f"Предупреждение: найдено только {len(class_samples)} классов из 10")

    # Создаем сетку 2x5 для отображения
    rows, cols = 2, 5
    cell_width, cell_height = 100, 100
    grid = np.zeros((rows * cell_height, cols * cell_width, 3), dtype=np.uint8) * 255

    for label in range(10):
        if label in class_samples:
            img_image = class_samples[label]
            img_np = np.array(img_image)
            img_resized = cv2.resize(img_np, (cell_width, cell_height), interpolation=cv2.INTER_LINEAR)
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
            row = label // cols
            col = label % cols

            grid[row * cell_height:(row + 1) * cell_height,
            col * cell_width:(col + 1) * cell_width] = img_resized

            cv2.putText(grid, f"Class {label}",
                        (col * cell_width + 5, row * cell_height + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

    cv2.imshow("MNIST - One example per class", grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite(f'{dataset_path}/mnist.png', grid)


if __name__ == '__main__':
    root_path = Path(__file__).parent.parent.parent.parent

    parser = argparse.ArgumentParser(description='Get PyTorch MNIST dataset')
    parser.add_argument('--path', type=str, default='data/datasets/mnist', help='dataset path')
    parser.add_argument('--download', type=bool, default=False, help='download flag')
    args = parser.parse_args()

    dataset_path = args.path
    if not os.path.isabs(dataset_path):
        dataset_path = os.path.join(root_path, Path(dataset_path))
    Path(dataset_path).mkdir(parents=True, exist_ok=True)
    print(dataset_path)

    download_flag = args.download

    train_dataset = MNISTDataset(dataset_path, train=True, download=download_flag)
    test_dataset = MNISTDataset(dataset_path, train=False, download=download_flag)

    show_one_per_class(train_dataset, dataset_path)
