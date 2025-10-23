import kagglehub
import shutil
from pathlib import Path

# Download dataset from kagglehub
path = kagglehub.dataset_download("grassknoted/asl-alphabet")
print("Path to dataset files:", path)

# Define raw data directory
raw_dir = Path(r"C:\Projects\ASL-Translator\data\raw")
raw_dir.mkdir(parents=True, exist_ok=True)

# Move dataset into raw_dir
dst = raw_dir / "asl_alphabet"
if dst.exists():
    print(f"{dst} already exists.")
else:
    shutil.move(path, dst)
    print("Moved dataset to {dst}")