import os
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class RAFDBDataset(Dataset):
    """Custom dataset class for RAF-DB facial expression dataset."""

    EMOTION_LABELS = {
        1: "Surprise",
        2: "Fear",
        3: "Disgust",
        4: "Happiness",
        5: "Sadness",
        6: "Anger",
        7: "Neutral",
    }

    def __init__(
        self,
        root_dir: str,
        split: Literal["train", "test"] = "train",
        transform: Callable | None = None,
        use_aligned: bool = False,
    ) -> None:
        """Initialize RAF-DB dataset

        Args:
            root_dir (str): Root directory of the RAF-DB dataset
            split (Literal["train", "test"]): Dataset split, either "train" or "test".
            transform (Callable | None): A function/transform to apply to the images.
            use_aligned (bool): Whether to use aligned images.
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.use_aligned = use_aligned

        self.dataset_dir = self.root_dir / "DATASET"
        self.image_dir = self.dataset_dir / split

        # Load labels
        if split == "train":
            label_file = self.dataset_dir / "train_labels.csv"
        else:
            label_file = self.dataset_dir / "test_labels.csv"

        if not label_file.exists():
            raise FileNotFoundError(f"Label file not found: {label_file}")

        self.labels_df = pd.read_csv(label_file)

        # Create image path to label mapping
        self.samples = []
        self._load_samples()

        print(f"Loaded {len(self.samples)} samples for {split} split.")
        self._print_class_distribution()

    def _load_samples(self) -> None:
        """Load image paths and labels."""

        for idx, row in self.labels_df.iterrows():
            if "image" in self.labels_df.columns:
                filename = row["image"]
            else:
                # If no header, assume first column is filename
                filename = row[0]

            if "label" in self.labels_df.columns:
                label = row["label"]
            else:
                # If no header, assume second column is label
                label = row[1]

            # Find image in subdirectories
            image_path = None

            for emotion_folder in range(1, 8):
                potential_path = self.image_dir / str(emotion_folder) / filename
                if potential_path.exists():
                    image_path = potential_path
                    break

            if image_path is None or not image_path.exists():
                print(f"Warning: Image not found: {filename}")
                continue

            # Convert label to 0-indexed TODO: Not sure if this is necessary
            label = int(label) - 1

            self.samples.append(
                {
                    "image_path": image_path,
                    "label": label,
                    "filename": filename,
                }
            )

    def _print_class_distribution(self) -> None:
        """Print distribution of emotions in dataset."""

        label_counts = {}

        for sample in self.samples:
            label = sample["label"]
            label_counts[label] = label_counts.get(label, 0) + 1

        print(f"\nClass distribution for {self.split} split:")
        for label_idx in sorted(label_counts.keys()):
            emotion_name = self.EMOTION_LABELS.get(label_idx + 1, "Unknown")
            count = label_counts[label_idx]
            percentage = (count / len(self.samples)) * 100
            print(f"  {emotion_name}: {count} ({percentage:.1f}%)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Get item by index

        Args:
            idx (int): Index

        Returns:
            tuple[torch.Tensor, int]: (image, label)
        """

        sample = self.samples[idx]
        image_path = sample["image_path"]
        label = sample["label"]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalance dataset

        Returns:
            torch.Tensor: Class weights
        """

        label_counts = {}

        for sample in self.samples:
            label = sample["label"]
            label_counts[label] = label_counts.get(label, 0) + 1

        total = len(self.samples)
        num_classes = len(self.EMOTION_LABELS)

        # Calculate weights: total / (num_classes * count)
        weights = []
        for i in range(num_classes):
            count = label_counts.get(i, 0)
            weight = total / (num_classes * count)
            weights.append(weight)

        return torch.FloatTensor(weights)
