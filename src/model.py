import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from sklearn.metrics import classification_report, confusion_matrix
from torchmetrics import Accuracy, ConfusionMatrix


class EmotionRecognitionModel(pl.LightningModule):
    """Emotion recognition model using EfficientNet-B0 as backbone."""

    EMOTION_LABELS = {
        0: "Surprise",
        1: "Fear",
        2: "Disgust",
        3: "Happiness",
        4: "Sadness",
        5: "Anger",
        6: "Neutral",
    }

    def __init__(
        self,
        num_classes: int = 7,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        dropout: float = 0.3,
        class_weights: torch.Tensor | None = None,
    ):
        """Initialize Model.

        Args:
            num_classes (int): Number of emotion classes.
            learning_rate (float): Learning rate for the optimizer.
            weight_decay (float): Weight decay for the optimizer.
            dropout (float): Dropout rate for the model.
            class_weights (torch.Tensor): Class weights for handling class imbalance.
        """

        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters(ignore=["class_weights"])

        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Load EfficientNet-B0 backbone (use weights instead of pretrained)
        self.backbone = models.efficientnet_b0(weights="DEFAULT")
        num_features = self.backbone.classifier[1].in_features

        # Replace classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True), nn.Linear(num_features, num_classes)
        )

        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Metrics
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

        # For confusion matrix
        self.test_predictions = []
        self.test_targets = []

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = self.train_accuracy(preds, labels)

        # Log metrics
        self.log(
            "train/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "train/acc",
            acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = self.val_accuracy(preds, labels)

        # Log metrics
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "val/acc", acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True
        )

        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = self.test_accuracy(preds, labels)

        # Store predictions for confusion matrix
        self.test_predictions.append(preds)
        self.test_targets.append(labels)

        # Log metrics
        self.log(
            "test/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "test/acc", acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True
        )

        return loss

    def on_test_epoch_end(self):
        """Called at the end of test epoch to compute confusion matrix."""

        # Concatenate all predictions and targets
        all_preds = torch.cat(self.test_predictions)
        all_targets = torch.cat(self.test_targets)

        # Compute confusion matrix
        self.plot_confusion_matrix(all_targets.cpu(), all_preds.cpu())

        # Clear stored predictions and targets
        self.test_predictions.clear()
        self.test_targets.clear()

    def plot_confusion_matrix(self, targets, preds):
        """Plot and save the confusion matrix, and log to WandB."""
        import wandb

        cm = confusion_matrix(targets, preds)

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=list(self.EMOTION_LABELS.values()),
            yticklabels=list(self.EMOTION_LABELS.values()),
        )
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()

        # Save locally
        plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")

        # Log to WandB if logger is available
        if self.logger:
            self.logger.experiment.log(
                {"confusion_matrix": wandb.Image("confusion_matrix.png")}
            )

        plt.close()

        print("\nConfusion matrix saved to confusion_matrix.png")

        # Get classification report
        report = classification_report(
            targets,
            preds,
            target_names=list(self.EMOTION_LABELS.values()),
            output_dict=True,
        )

        print("\nClassification Report:")
        print(
            classification_report(
                targets,
                preds,
                target_names=list(self.EMOTION_LABELS.values()),
            )
        )

        # Log per-class metrics to WandB
        if self.logger:
            for emotion, metrics in report.items():
                if emotion in self.EMOTION_LABELS.values():
                    self.logger.experiment.log(
                        {
                            f"test/{emotion}_precision": metrics["precision"],
                            f"test/{emotion}_recall": metrics["recall"],
                            f"test/{emotion}_f1": metrics["f1-score"],
                        }
                    )

            # Log overall metrics
            self.logger.experiment.log(
                {
                    "test/macro_avg_f1": report["macro avg"]["f1-score"],
                    "test/weighted_avg_f1": report["weighted avg"]["f1-score"],
                }
            )

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""

        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=0.5,
                patience=3,
            ),
            "monitor": "val/acc",
            "interval": "epoch",
            "frequency": 1,
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}


if __name__ == "__main__":
    # Test model
    model = EmotionRecognitionModel(num_classes=7, learning_rate=1e-3, dropout=0.3)

    # Test forward pass
    dummy_input = torch.randn(4, 3, 224, 224)
    output = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
