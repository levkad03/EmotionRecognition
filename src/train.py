import albumentations as A
import pytorch_lightning as pl
import torch
from albumentations.pytorch import ToTensorV2
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from src.dataset import RAFDBDataset
from src.model import EmotionRecognitionModel


def get_transforms():
    """Get train and test transforms."""

    train_transform = A.Compose(
        [
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.Affine(
                translate_percent=0.1, scale=(0.9, 1.1), p=0.5
            ),  # Replaces ShiftScaleRotate
            A.CoarseDropout(
                num_holes_range=(4, 8),
                hole_height_range=(8, 16),
                hole_width_range=(8, 16),
                p=0.3,
            ),
            A.GaussNoise(var_limit=(10.0, 50.0), mean=0, p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )

    test_transform = A.Compose(
        [
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )

    return train_transform, test_transform


def main():

    # Set matmul precision for better performance
    torch.set_float32_matmul_precision("medium")

    # Hyperparameters
    BATCH_SIZE = 16
    NUM_WORKERS = 4
    MAX_EPOCHS = 20
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    DROPOUT = 0.3

    # Get transforms
    train_transform, test_transform = get_transforms()

    # Create datasets
    print("Loading datasets...")
    train_dataset = RAFDBDataset(
        root_dir="data",
        split="train",
        transform=train_transform,
    )

    test_dataset = RAFDBDataset(
        root_dir="data",
        split="test",
        transform=test_transform,
    )

    # Get class weights
    class_weights = train_dataset.get_class_weights()
    print(f"\nClass weights: {class_weights}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True if NUM_WORKERS > 0 else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True if NUM_WORKERS > 0 else False,
    )

    model = EmotionRecognitionModel(
        num_classes=7,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        dropout=DROPOUT,
        class_weights=class_weights,
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="emotion-{epoch:02d}-{val/acc:.4f}",
        monitor="val/acc",
        mode="max",
        save_top_k=3,
        save_last=True,
        verbose=True,
    )

    early_stopping_callback = EarlyStopping(
        monitor="val/acc",
        patience=10,
        mode="max",
        verbose=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    logger = WandbLogger(
        project="emotion-recognition",
        name="efficientnet-b0",
        log_model=True,
    )

    # Log hyperparameters to W&B
    logger.experiment.config.update(
        {
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "dropout": DROPOUT,
            "max_epochs": MAX_EPOCHS,
            "architecture": "EfficientNet-B0",
            "dataset": "RAF-DB",
        }
    )

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping_callback, lr_monitor],
        log_every_n_steps=10,
        precision="16-mixed" if torch.cuda.is_available() else 32,
    )

    print("\nStarting training...")
    trainer.fit(model, train_loader, test_loader)

    print("\nRunning final test...")
    trainer.test(model, test_loader, ckpt_path="best")

    print("\nTraining complete!")
    print(f"Best model saved in: {checkpoint_callback.best_model_path}")
    print(f"\nView results on WandB: {logger.experiment.url}")


if __name__ == "__main__":
    main()
