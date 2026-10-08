import os

# Hugging Face mirror for environments where huggingface.co is slow/unreachable
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from pathlib import Path
from shutil import copy2

from huggingface_hub import hf_hub_download

REPO_ID = "netop/gotsf-ds"
REPO_TYPE = "dataset"

FILES = [
    "data/train/DLPRB_train_0w-5w.csv",
    "data/train/DLThpVol_train_0w-5w.csv",
    "data/train/DLThpTime_train_0w-5w.csv",
    "data/train/MR_number_train_0w-5w.csv",
]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for repo_path in FILES:
        print(f"Downloading {repo_path} ...")

        cached_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=repo_path,
            repo_type=REPO_TYPE,
        )

        destination = RAW_DIR / Path(repo_path).name
        copy2(cached_path, destination)

        print(f"Saved to: {destination}")

    print("\nAll KPI files downloaded successfully.")


if __name__ == "__main__":
    main()