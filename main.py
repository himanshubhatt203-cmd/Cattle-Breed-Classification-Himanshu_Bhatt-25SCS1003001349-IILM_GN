import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("No CUDA GPU found!")
from src.dataset import get_dataloaders
from src.model import get_model
from src.train import train_model
from src.evaluate import evaluate
from pathlib import Path

# Set your paths
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"

train_dir = DATASET_DIR / "train"
val_dir = DATASET_DIR / "val"
test_dir = DATASET_DIR / "test"
batch_size = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data
train_loader, val_loader, test_loader, class_names, num_classes = get_dataloaders(
    train_dir, val_dir, test_dir, batch_size
)

# Model
model = get_model(num_classes, device)

# Train
train_model(model, train_loader, val_loader, device, num_epochs=100)

# Evaluate
evaluate(model, test_loader, class_names, device)
