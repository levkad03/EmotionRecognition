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

        # Load EfficientNet-B0 backbone
        self.backbone = models.efficientnet_b0(pretrained=True)
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
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", acc, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = self.val_accuracy(preds, labels)

        # Log metrics
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

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
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

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
        """Plot and save the confusion matrix."""

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
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
        plt.close()

        print("\nConfusion matrix saved to confusion_matrix.png")

        print("\nClassification Report:")
        print(
            classification_report(
                targets,
                preds,
                target_names=list(self.EMOTION_LABELS.values()),
            )
        )

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=0.5,
                patience=3,
                verbose=True,
            ),
            "monitor": "val/acc",
            "interval": "epoch",
            "frequency": 1,
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
