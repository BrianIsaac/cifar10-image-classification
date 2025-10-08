# Standard library
import os
from pathlib import Path

# Third-party libraries
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm

# Configuration
output_dir = "data/cifar10_resized_224"
os.makedirs(output_dir, exist_ok=True)

# Define the resize transform
resize_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load CIFAR-10 dataset (train and test)
datasets_to_process = {
    "train": datasets.CIFAR10(root="data", train=True, download=True),
    "test": datasets.CIFAR10(root="data", train=False, download=True),
}

# Create class subfolders directly under output_dir
dataset_classes = datasets_to_process["train"].classes
for class_name in dataset_classes:
    class_dir = Path(output_dir) / class_name
    class_dir.mkdir(parents=True, exist_ok=True)

# Process and save resized images into class folders under output_dir
for split_name, dataset in datasets_to_process.items():
    for idx in tqdm(range(len(dataset)), desc=f"Processing {split_name} set"):
        img, label = dataset[idx]
        img_resized = resize_transform(img)

        # Define output path (no train/test split folders, only class subfolders)
        class_name = dataset.classes[label]
        class_dir = Path(output_dir) / class_name
        img_path = class_dir / f"{split_name}_{idx:05d}.png"

        # Save image
        save_image(img_resized, img_path)

print("Preprocessing completed. Resized images saved in:", output_dir)
