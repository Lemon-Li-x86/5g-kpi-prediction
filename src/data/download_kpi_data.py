"""Download the four fixed gotsf-ds KPI source CSVs to data/raw/.

Uses the public Hugging Face dataset; HF_ENDPOINT is set only as a default
and can be overridden by the user's environment.
"""

import os

# Hugging Face mirror for environments where huggingface.co is slow/unreachable
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from pathlib import Path
from shutil import copy2

from huggingface_hub import hf_hub_download

from src.core.config import RAW_DATA, DATASET_REPO_ID, DATASET_FILE_DIRECTORY, DATASET_FILES

REPO_ID = DATASET_REPO_ID
REPO_TYPE = "dataset"
FILES = [f"{DATASET_FILE_DIRECTORY}/{filename}" for filename in DATASET_FILES.values()]
RAW_DIR = RAW_DATA


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