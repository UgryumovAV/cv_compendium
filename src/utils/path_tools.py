import io
import os.path
import shutil
from pathlib import Path


def create_log_file(model_name: str) -> io.TextIOWrapper:
    root_path = Path(__file__).resolve(strict=True).parent.parent.parent
    log_file_dir_path = os.path.join(root_path, Path("data/logs"), Path(model_name), Path("log_file"))
    if os.path.exists(log_file_dir_path):
        shutil.rmtree(log_file_dir_path)
    Path(log_file_dir_path).mkdir(parents=True, exist_ok=True)
    return open(os.path.join(log_file_dir_path, f"log_{model_name}.txt"), "w")


def get_root():
    return Path(__file__).resolve(strict=True).parent.parent.parent.parent
