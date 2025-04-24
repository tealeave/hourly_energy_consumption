import kagglehub
from pathlib import Path
import shutil

# 1. Download (or update) the dataset; `path` is a pathlib.Path to the folder
path: Path = Path(kagglehub.dataset_download("robikscube/hourly-energy-consumption"))

# 2. Define the target parent data folder (../data/ relative to this file)
base_dir = Path(__file__).resolve().parent      # wherever this script lives
print(f"Base dir: {base_dir}")

# The data folder is one level up from the script and then in a folder called data
data_dir = base_dir.parent.joinpath("data")  # e.g. ../data
print(f"Data dir: {data_dir}")

# Create the data directory if it doesn't exist
data_dir.mkdir(parents=True, exist_ok=True)

# 3. Move the downloaded folder into ../data/
dest = data_dir / path.name     # e.g. data/hourly-energy-consumption

# If there’s already an old copy, remove it first:
if dest.exists():
    shutil.rmtree(dest)

shutil.move(str(path), str(dest))

print(f"Dataset is now at: {dest}")
