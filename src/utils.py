# Standard library
import os
import shutil
import json
from collections import Counter
from typing import Tuple, List, Optional, Dict, Any

# Third-party libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

# PyTorch core
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

# TorchVision
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torchvision.models as models
import torchvision.transforms as T

# Albumentations
import albumentations as A
from albumentations import Compose

# Metrics
from sklearn.metrics import classification_report

def load_dataset(dir: str) -> None:
    """
    Downloads CIFAR dataset into specified path.
    
    Parameters:
    - dir (str): Path to the directory to save CIFAR dataset
    """

    # Define the transform (basic ToTensor for EDA)
    transform = transforms.ToTensor()

    # Load training dataset
    train_dataset = datasets.CIFAR10(
        root=dir,
        train=True,
        download=True,
        transform=transform
    )

    # Load test dataset
    test_dataset = datasets.CIFAR10(
        root=dir,
        train=False,
        download=True,
        transform=transform
    )

    # Print dataset info
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of test samples: {len(test_dataset)}")
    print(f"Classes: {train_dataset.classes}")

class ImageEDA:
    def __init__(self, data_dir: str):
        """
        Initialise CIFAR10EDA with dataset path and load data.
        
        Parameters:
        - data_dir (str): Root directory for storing the dataset.
        """
        self.data_dir = data_dir
        self.transform = transforms.ToTensor()
        self.train_dataset = datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            download=True,
            transform=self.transform
        )
        self.test_dataset = datasets.CIFAR10(
            root=self.data_dir,
            train=False,
            download=True,
            transform=self.transform
        )
        self.class_names = self.train_dataset.classes
    
    def show_sample_images(self, samples_per_class: int = 5) -> None:
        """
        Display sample images for each class.
        
        Parameters:
        - samples_per_class (int): Number of images to display per class.
        """
        for cls_idx, cls_name in enumerate(self.class_names):
            indices = [i for i, (_, label) in enumerate(self.train_dataset) if label == cls_idx]
            fig, axes = plt.subplots(1, samples_per_class, figsize=(15, 3))
            fig.suptitle(f"Class: {cls_name}", fontsize=14)
            for ax, idx in zip(axes, indices[:samples_per_class]):
                image, _ = self.train_dataset[idx]
                img_np = image.permute(1, 2, 0).numpy()
                ax.imshow(img_np)
                ax.axis("off")
            plt.tight_layout()
            plt.show()
    
    def analyse_class_distribution(self) -> pd.DataFrame:
        """
        Compute and display the number of images per class.
        
        Returns:
        - pd.DataFrame: Class distribution summary.
        """
        labels = [label for _, label in self.train_dataset]
        counts = Counter(labels)
        df = pd.DataFrame({
            "class": [self.class_names[i] for i in counts.keys()],
            "count": list(counts.values())
        }).sort_values(by="count", ascending=False).reset_index(drop=True)
        
        # Plot class distribution
        plt.figure(figsize=(10, 5))
        plt.bar(df["class"], df["count"], color="skyblue")
        plt.title("Class Distribution in CIFAR-10")
        plt.ylabel("Number of Images")
        plt.xticks(rotation=45)
        plt.show()
        
        return df
    
    def examine_image_properties(self) -> None:
        """
        Examine image dimensions and color channel consistency.
        """
        image, _ = self.train_dataset[0]
        print(f"Sample image shape: {image.shape} (C, H, W)")
        print(f" - Channels: {image.shape[0]}")
        print(f" - Height: {image.shape[1]}, Width: {image.shape[2]}")
        print("\nAll images in CIFAR-10 are expected to be 3x32x32 (RGB, 32x32 pixels).")

    def plot_locationwise_mean_std(self, class_name: str, num_samples: int = None) -> None:
        """
        Plot heatmaps of mean and std pixel intensities across samples for a class.

        Parameters:
        - class_name (str): Name of the class to analyse.
        - num_samples (int or None): Number of samples to include. If None, uses all.
        """
        if class_name not in self.class_names:
            raise ValueError("Invalid class name.")
        
        cls_idx = self.class_names.index(class_name)
        images = [self.train_dataset[i][0] for i in range(len(self.train_dataset)) if self.train_dataset[i][1] == cls_idx]

        if num_samples is not None:
            images = images[:num_samples]

        # Stack and convert to numpy: shape (N, C, H, W)
        img_stack = torch.stack(images).numpy()

        # Calculate mean and std across samples
        mean_image = img_stack.mean(axis=0)  # (C, H, W)
        std_image = img_stack.std(axis=0)    # (C, H, W)

        # Plot mean heatmap per channel
        plt.figure(figsize=(12, 6))
        for i, title in enumerate(['Red Mean', 'Green Mean', 'Blue Mean']):
            plt.subplot(2, 3, i+1)
            plt.imshow(mean_image[i], cmap='viridis')
            plt.title(f"{class_name} {title}")
            plt.axis('off')
            plt.colorbar()

        # Plot std heatmap per channel
        for i, title in enumerate(['Red Std', 'Green Std', 'Blue Std']):
            plt.subplot(2, 3, i+4)
            plt.imshow(std_image[i], cmap='magma')
            plt.title(f"{class_name} {title}")
            plt.axis('off')
            plt.colorbar()

        plt.tight_layout()
        plt.show()

class AlbumentationsDataset(Dataset):
    """
    A PyTorch Dataset wrapper for Albumentations transforms.

    Converts PIL images to NumPy arrays, applies Albumentations transformations,
    and returns image tensors and class labels.
    """

    def __init__(self, dataset: Dataset, transform: Compose) -> None:
        """
        Parameters:
        - dataset (Dataset): A PyTorch dataset (e.g., a subset of ImageFolder).
        - transform (Compose): An Albumentations Compose transform.
        """
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get one transformed image-label pair.

        Returns:
        - Tuple[torch.Tensor, int]: Transformed image and corresponding class label.
        """
        image, label = self.dataset[idx]

        # Standardise to numpy array
        if hasattr(image, "convert"):  # PIL Image
            image = image.convert("RGB")
            image = np.array(image)
        else:  # Tensor
            image = image.permute(1, 2, 0).numpy()

        augmented = self.transform(image=image)
        return augmented["image"], label

    def __len__(self) -> int:
        """
        Get the total number of samples in the dataset.
        """
        return len(self.dataset)

class ImageLoader:
    """
    A modular image data loader for classification, supporting CIFAR-10 or preprocessed ImageFolder datasets
    with Albumentations preprocessing and stratified splitting.
    """

    def __init__(
        self,
        dataset_name: str,
        data_dir: str,
        train_transform: A.Compose,
        val_transform: Optional[A.Compose] = None,
        batch_size: int = 32,
        train_split: float = 0.8,
        image_size: int = 32,
        seed: int = 42,
        num_workers: int = 4
    ) -> None:
        """
        Initialise the ImageLoader with a specific dataset and loading configuration.

        Parameters:
        - dataset_name (str): Name of the dataset to load ('cifar10' or 'imagefolder').
        - data_dir (str): Root directory for storing or locating the dataset.
        - train_transform (A.Compose): Albumentations transform for training.
        - val_transform (Optional[A.Compose]): Albumentations transform for validation. Defaults to train_transform.
        - batch_size (int): Number of samples per batch.
        - train_split (float): Proportion of data used for training (0 < train_split < 1).
        - image_size (int): Expected image size. Loads from ImageFolder if 224, CIFAR-10 if 32.
        - seed (int): Random seed for reproducibility.
        - num_workers (int): Number of worker threads for data loading.
        """
        self.batch_size = batch_size
        self.train_split = train_split
        self.seed = seed
        self.num_workers = num_workers
        self.image_size = image_size

        if image_size == 32:
            # Load CIFAR-10 dataset
            print("Loading CIFAR-10 dataset (32x32)...")
            base_dataset = datasets.CIFAR10(
                root=data_dir,
                train=True,
                download=True,
                transform=None  # Albumentations will handle transforms
            )
            self.class_names = getattr(base_dataset, 'classes', None)

        elif image_size == 224:
            # Load from preprocessed ImageFolder dataset
            print("Loading preprocessed ImageFolder dataset (224x224)...")
            resized_folder = os.path.join(data_dir, "cifar10_resized_224")

            # Check if folder exists, if not run preprocess.py
            if not os.path.exists(resized_folder):
                print(f"{resized_folder} not found. Running preprocessing script...")
                import subprocess
                subprocess.run(["python", "preprocess.py"], check=True)
                print("Preprocessing completed.")

            # Load dataset after ensuring it exists
            base_dataset = datasets.ImageFolder(
                root=resized_folder,
            )
            self.class_names = base_dataset.classes

        else:
            print(f"Unsupported image size: {image_size}. Only 32 or 224 supported.")
            raise ValueError(f"Unsupported image size: {image_size}. Only 32 or 224 supported.")

        # Perform stratified split
        train_set, val_set = self._stratified_split(base_dataset)

        # Use provided transforms directly (no resizing prepended)
        self.train_transform = train_transform
        self.val_transform = val_transform if val_transform else train_transform

        # Wrap with AlbumentationsDataset
        self.train_dataset = AlbumentationsDataset(train_set, self.train_transform)
        self.val_dataset = AlbumentationsDataset(val_set, self.val_transform)

        # Create DataLoaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

    def _stratified_split(self, dataset: Dataset) -> Tuple[Subset, Subset]:
        """
        Internal method to perform stratified train-validation split.

        Returns:
        - Tuple[Subset, Subset]: train and validation subsets
        """
        if hasattr(dataset, 'targets'):
            targets = dataset.targets
        elif hasattr(dataset, 'labels'):
            targets = dataset.labels
        else:
            raise AttributeError("Dataset must have 'targets' or 'labels' attribute for stratified split.")

        sss = StratifiedShuffleSplit(n_splits=1, train_size=self.train_split, random_state=self.seed)
        train_idx, val_idx = next(sss.split(X=targets, y=targets))
        return Subset(dataset, train_idx), Subset(dataset, val_idx)

    def get_class_names(self) -> Optional[List[str]]:
        """
        Returns a list of the class names if available.

        Returns:
        - Optional[List[str]]: Class names or None.
        """
        return self.class_names

    def get_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Get the training and validation DataLoaders.

        Returns:
        - Tuple[DataLoader, DataLoader]: train_loader and val_loader
        """
        return self.train_loader, self.val_loader

    def get_datasets(self) -> Tuple[Dataset, Dataset]:
        """
        Get the wrapped Albumentations-compatible datasets.

        Returns:
        - Tuple[Dataset, Dataset]: train_dataset and val_dataset
        """
        return self.train_dataset, self.val_dataset
    
class CIFARClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        base_model_name: str = "resnet18",
        freeze_backbone: bool = False,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ) -> None:
        """
        CIFARClassifier with ResNet18, DenseNet121, Vision Transformer, or GoogLeNet backbones,
        using a deeper classifier head.

        Parameters:
        - num_classes (int): Number of output classes.
        - base_model_name (str): Backbone architecture ('resnet18', 'densenet121', 'vit_b_16', 'googlenet').
        - freeze_backbone (bool): Whether to freeze backbone weights during training.
        - device (torch.device): Device to place the model on.
        """
        super().__init__()
        self.base_model_name = base_model_name.lower()
        self.device = device

        def deep_fc_head(in_features, num_classes):
            return nn.Sequential(
                nn.Linear(in_features, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Dropout(0.4),

                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.3),

                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.2),

                nn.Linear(256, num_classes)
            )

        if self.base_model_name == "resnet18":
            self.base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            in_features = self.base_model.fc.in_features
            self.base_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.base_model.maxpool = nn.Identity()
            self.base_model.fc = deep_fc_head(in_features, num_classes)

            if freeze_backbone:
                for name, param in self.base_model.named_parameters():
                    if not name.startswith("fc"):
                        param.requires_grad = False

        elif self.base_model_name == "densenet121":
            self.base_model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
            in_features = self.base_model.classifier.in_features
            self.base_model.classifier = deep_fc_head(in_features, num_classes)

            if freeze_backbone:
                for name, param in self.base_model.named_parameters():
                    if not name.startswith("classifier"):
                        param.requires_grad = False

        elif self.base_model_name == "vit_b_16":
            self.base_model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
            in_features = self.base_model.heads.head.in_features
            self.base_model.heads.head = deep_fc_head(in_features, num_classes)

            if freeze_backbone:
                for name, param in self.base_model.named_parameters():
                    if not name.startswith("heads.head"):
                        param.requires_grad = False

        elif self.base_model_name == "googlenet":
            self.base_model = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT, aux_logits=True)
            in_features = self.base_model.fc.in_features
            self.base_model.fc = deep_fc_head(in_features, num_classes)

            if freeze_backbone:
                for name, param in self.base_model.named_parameters():
                    if not name.startswith("fc"):
                        param.requires_grad = False

        else:
            raise ValueError(f"Unsupported base model: {base_model_name}")

        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base_model(x)

        # Handle special output cases (e.g. GoogLeNet)
        if hasattr(out, "logits"):  # NamedTuple from GoogLeNet with aux_logits=True
            return out.logits

        return out

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        lr: float = 1e-3,
        epochs: int = 10,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> None:
        """
        Trainer class to handle model training and validation.

        Parameters:
        - model (nn.Module): The neural network model to train.
        - train_loader (DataLoader): DataLoader for training data.
        - val_loader (DataLoader): DataLoader for validation data.
        - device (torch.device): Device to train on. Uses GPU if it is available, else CPU.
        - lr (float): Learning rate.
        - epochs (int): Number of training epochs.
        - criterion (nn.Module): Loss function. Defaults to CrossEntropyLoss.
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.optimizer = optimizer or torch.optim.Adam(self.model.parameters(), lr=lr)

        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.train_accuracies: List[float] = []
        self.val_accuracies: List[float] = []

    def _compute_accuracy(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
        preds = outputs.argmax(dim=1)
        correct = (preds == targets).sum().item()
        return correct / targets.size(0)
    
    def optimizer_state_dict(self) -> Dict[str, Any]:
        """
        Return the state dictionary of the optimizer.

        Returns:
        - Dict[str, Any]: Optimizer state dict.
        """
        return self.optimizer.state_dict()

    def train(self) -> None:
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            running_accuracy = 0.0

            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                running_accuracy += self._compute_accuracy(outputs, labels)

            train_loss = running_loss / len(self.train_loader)
            train_acc = running_accuracy / len(self.train_loader)

            val_loss, val_acc = self.evaluate()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)

            print(
                f"Epoch [{epoch+1}/{self.epochs}] "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )

    def train_one_epoch(self, use_tqdm: bool = False, scheduler=None) -> Tuple[float, float]:
        self.model.train()
        running_loss = 0.0
        running_accuracy = 0.0

        data_iter = self.train_loader
        if use_tqdm:
            data_iter = tqdm(self.train_loader, desc="Training Batches")

        for images, labels in data_iter:
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            if scheduler is not None:
                scheduler.step()

            running_loss += loss.item()
            running_accuracy += self._compute_accuracy(outputs, labels)

        train_loss = running_loss / len(self.train_loader)
        train_acc = running_accuracy / len(self.train_loader)

        val_loss, val_acc = self.evaluate(use_tqdm=True)

        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accuracies.append(train_acc)
        self.val_accuracies.append(val_acc)

        print(
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        return train_loss, train_acc

    def evaluate(self, use_tqdm: bool = False) -> tuple:
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0

        data_iter = self.val_loader
        if use_tqdm:
            data_iter = tqdm(self.val_loader, desc="Validation Batches")

        with torch.no_grad():
            for images, labels in data_iter:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                total_accuracy += self._compute_accuracy(outputs, labels)

        avg_loss = total_loss / len(self.val_loader)
        avg_accuracy = total_accuracy / len(self.val_loader)
        return avg_loss, avg_accuracy
    
    def plot_metrics(self) -> None:
        """
        Plot training and validation loss and accuracy over epochs.
        """
        epochs = range(1, self.epochs + 1)

        plt.figure(figsize=(14, 5))

        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_losses, label='Train Loss')
        plt.plot(epochs, self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss over Epochs')
        plt.legend()
        plt.grid(True)

        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.train_accuracies, label='Train Accuracy')
        plt.plot(epochs, self.val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy over Epochs')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def final_evaluation(self) -> dict:
        """
        Compute and print final macro-averaged F1 score, precision, and recall on the validation set.
        Returns the classification report as a dict.
        """
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(labels.cpu().numpy())

        # Generate full classification report as dict
        report = classification_report(all_targets, all_preds, zero_division=0, output_dict=True)

        # Print summary metrics
        print("Final Validation Evaluation:")
        print(f"Precision (macro): {report['macro avg']['precision']:.4f}")
        print(f"Recall (macro):    {report['macro avg']['recall']:.4f}")
        print(f"F1 Score (macro):  {report['macro avg']['f1-score']:.4f}")

        return report

    def save_misclassified(self, output_dir: str, class_names: List[str], max_images: int = 10) -> dict:
        """
        Save misclassified validation images and metadata.

        Parameters:
        - output_dir (str): Directory to save images and metadata JSON.
        - class_names (List[str]): List of class names, indexed by label.
        - max_images (int): Maximum number of misclassified images to save.

        Returns:
        - dict: Metadata containing filenames, true labels, and predicted labels.
        """
        # Clear existing directory if it exists
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        self.model.eval()
        metadata = []

        # Inverse normalisation for saving
        inv_normalize = T.Compose([
            T.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
            T.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])
        ])

        with torch.no_grad():
            idx_counter = 0
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                preds = outputs.argmax(dim=1)

                for img, pred, true in zip(images, preds, labels):
                    if pred != true:
                        img_cpu = img.cpu()
                        img_inv = inv_normalize(img_cpu).clamp(0,1)
                        filename = f"misclassified_{idx_counter:03d}_true-{class_names[true]}_pred-{class_names[pred]}.png"
                        filepath = os.path.join(output_dir, filename)
                        save_image(img_inv, filepath)

                        metadata.append({
                            "filename": filename,
                            "true_label": class_names[true],
                            "predicted_label": class_names[pred]
                        })

                        idx_counter += 1
                        if idx_counter >= max_images:
                            break
                if idx_counter >= max_images:
                    break

        # Save metadata as JSON
        metadata_path = os.path.join(output_dir, "misclassified_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved {len(metadata)} misclassified images and metadata to {output_dir}")
        return metadata