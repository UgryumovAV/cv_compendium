from collections import Counter
from typing import Any

import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler


class BalancedDataLoader:
    def __init__(
            self,
            dataset: Any,
            batch_size: int
    ):
        self.dataset = dataset
        self.batch_size = batch_size

        self.labels = self._extract_labels()
        self.sampler = self._create_sampler()
        self.dataloader = self._create_dataloader()

    def _extract_labels(self):
        labels = list()
        for i in range(len(self.dataset)):
            if hasattr(self.dataset, 'targets'):
                labels.append(self.dataset.targets[i])
            else:
                _, label = self.dataset[i]
                labels.append(label)
        return np.array(labels)

    def _create_sampler(self):
        class_counts = Counter(self.labels)
        class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
        sample_weights = [class_weights[label] for label in self.labels]

        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

    def _create_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=self.sampler,
            pin_memory=True,
            num_workers=8
        )

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)
