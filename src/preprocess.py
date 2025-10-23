import os
import cv2
import random
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torchvision import transforms
from PIL import Image

# define paths
RAW_DIR = Path(r"C:\Projects\ASL-Translator\data\raw\asl_alphabet\asl_alphabet_train\asl_alphabet_train")
PROCESSED_DIR = Path(r"C:\Projects\ASL-Translator\data\processed")
SPLIT_DIR = Path(r"C:\Projects\ASL-Translator\data\splits")

IMG_SIZE = 128  # resize images to 128x128

# augmentations
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor()
])

def preprocess_and_save():
    classes = sorted([d for d in os.listdir(RAW_DIR) if os.path.isdir(RAW_DIR / d)])
    print(f"Found {len(classes)} classes")
    
    all_data = []
    all_labels = []
    
    for label, cls in enumerate(tqdm(classes)): # labels: 0-28; classes in order: A-D, del, E-N, nothing, O-S, space, T-Z
        img_dir = RAW_DIR / cls
        images = os.listdir(img_dir)
        
        for img_name in images:
            img_path = img_dir / img_name
            try:
                img = Image.open(img_path).convert("RGB")
                img_t = transform(img)
                all_data.append(img_t.numpy())
                all_labels.append(label)
            except Exception as e:
                print(f"Error with {img_path}: {e}")
    
    all_data = np.array(all_data)
    all_labels = np.array(all_labels)
    
    # split into train/val/test
    X_train, X_temp, y_train, y_temp = train_test_split(all_data, all_labels, test_size=0.3, stratify=all_labels, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
    
    # save data
    np.savez_compressed(PROCESSED_DIR / "train.npz", X=X_train, y=y_train)
    np.savez_compressed(PROCESSED_DIR / "val.npz", X=X_val, y=y_val)
    np.savez_compressed(PROCESSED_DIR / "test.npz", X=X_test, y=y_test)
    
    print(f"Saved processed data to {PROCESSED_DIR}")

if __name__ == "__main__":
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    preprocess_and_save()