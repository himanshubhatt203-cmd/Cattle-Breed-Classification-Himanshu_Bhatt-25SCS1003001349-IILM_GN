import os
import json

DATASET_DIR = r"C:\Users\himan\Downloads\AI Model Data\Processed Data"
OUTPUT_PATH = r"C:\Users\himan\Downloads\Cattle_Breed_Classification\models\labels.json"

# Check if dataset folder exists
if not os.path.exists(DATASET_DIR):
    raise FileNotFoundError(f"Dataset path not found: {DATASET_DIR}")

# Get class names (folder names inside 'train')
class_names = sorted(os.listdir(DATASET_DIR))
print(f"Found {len(class_names)} classes:")
for idx, name in enumerate(class_names):
    print(f"{idx}: {name}")

# Create mapping dictionary
labels_dict = {idx: name for idx, name in enumerate(class_names)}

# Save to JSON
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump(labels_dict, f, indent=4)

print(f"\nlabels.json created successfully at:\n{OUTPUT_PATH}")