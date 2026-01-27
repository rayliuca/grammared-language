from grammared_language.triton.builder.repo_builder import TritonRepoBuilder
from grammared_language.utils import config_parser
import os
import logging
from pathlib import Path

DEFAULT_MODEL_CONFIG_PATH = "/default_model_config.yaml"
MODEL_CONFIG_PATH = "/model_config.yaml"
MODEL_REPO_FOLDER = "/models"

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
    Path(MODEL_REPO_FOLDER).mkdir(parents=True, exist_ok=True)
    env_vars = {k: v for k, v in os.environ.items() if k.startswith("GRAMMARED_LANGUAGE__")}
    # Determine which config path to use
    if os.path.isfile(MODEL_CONFIG_PATH):
        config = config_parser.load_config_from_file(MODEL_CONFIG_PATH)
    elif env_vars:
        config = config_parser.load_config_from_env()
    else:
        config = config_parser.load_config_from_file(DEFAULT_MODEL_CONFIG_PATH)

    # Build the model repository
    builder = TritonRepoBuilder()
    builder.build_model_repo(
        repo_folder=MODEL_REPO_FOLDER,
        config=config
    )


if __name__ == "__main__":
    build_triton_model_repo()