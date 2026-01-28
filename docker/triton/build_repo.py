from grammared_language.triton.builder.repo_builder import TritonRepoBuilder
from grammared_language.utils.config_parser import get_config, MODEL_REPO_FOLDER
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


"""
1. check if MODEL_REPO_FOLDER exists
2. check MODEL_CONFIG_PATH
3. check environment variable
4. use DEFAULT_MODEL_CONFIG_PATH
"""


def is_non_empty_dir(directory_path):
    """
    Checks if the directory exists, is a directory, and is not empty.
    """
    if not os.path.isdir(directory_path):
        return False
    
    # Check if directory is empty using os.scandir()
    try:
        with os.scandir(directory_path) as it:
            return any(it) # Returns True if there is any entry, False otherwise
    except OSError:
        # Handle potential permission errors or other OS issues
        return False
    

def build_triton_model_repo():
    if is_non_empty_dir(MODEL_REPO_FOLDER):
        logger.info(f"Model repository folder '{MODEL_REPO_FOLDER}' already exists and is not empty. Skipping build.")
        return
    model_repo_path = os.environ.get("GRAMMARED_LANGUAGE__MODEL_REPO_FOLDER", MODEL_REPO_FOLDER)
    config = get_config()
    # Build the model repository
    Path(model_repo_path).mkdir(parents=True, exist_ok=True)
    builder = TritonRepoBuilder()
    builder.build_model_repo(
        repo_folder=model_repo_path,
        config=config
    )


if __name__ == "__main__":
    build_triton_model_repo()