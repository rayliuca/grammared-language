# from .triton.triton_templates import
from jinja2 import Environment, FileSystemLoader
from pathlib import Path
import yaml
from .data_model import ModelsConfig

DEFAULT_TEMPLATE_FOLDER = str(Path(__file__).parent / "triton_templates")

SUPPORTED_MODEL_TYPES = ("gector", "grammared_classifier")

class TritonRepoBuilder:
    def __init__(self, template_folder: str|None=None):
        if template_folder is None:
            template_folder = DEFAULT_TEMPLATE_FOLDER
        self.template_folder = template_folder
        self.jina_loader = FileSystemLoader(searchpath=self.template_folder)
        self.jinja_env = Environment(loader=self.jina_loader)

    def load_config_from_file(self, file_name: str) -> ModelsConfig:
        """Load configuration from a YAML file.

        Args:
            file_name: Path to the YAML configuration file

        Returns:
            ModelsConfig: Parsed configuration object

        Raises:
            FileNotFoundError: If the configuration file doesn't exist
            yaml.YAMLError: If the YAML is invalid
            ValueError: If the configuration structure is invalid
        """
        config_path = Path(file_name)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_name}")

        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)

        # Filter out template/documentation entries (like 'name' in the example)
        # These are entries that contain union types in parentheses, indicating they're templates
        if 'models' not in config_data:
            config_data = {'models': config_data}

        models = config_data['models']
        triton_models = {'models': {}}
        for model_name, model_config in models.items():
            if model_config.backend == "triton":
                triton_models['models'][model_name] = model_config

        # Parse and validate with Pydantic
        config = ModelsConfig(**triton_models)
        return config

    def load_config_from_env(self) -> ModelsConfig:
        """Load configuration from environment variables.

        The environment variables should be set like:
        GRAMMARED_LANGUAGE__MODELS__<MODEL_NAME>__<FIELD>=<VALUE>

        note:
        - Use double underscores `__` to separate levels in the hierarchy.

        Example:
            GRAMMARED_LANGUAGE__MODELS__GECTOR_BERT__TYPE=gector
            GRAMMARED_LANGUAGE__MODELS__GECTOR_BERT__BACKEND=triton
            GRAMMARED_LANGUAGE__MODELS__GECTOR_BERT__PRETRAINED_MODEL_NAME_OR_PATH=grammarly/gector-bert-base-cased

        Returns:
            ModelsConfig: Parsed configuration object from environment variables
        """
        import os

        prefix = "GRAMMARED_LANGUAGE__MODELS__"
        models_dict = {}

        # Iterate through all environment variables
        for env_key, env_value in os.environ.items():
            if not env_key.startswith(prefix):
                continue

            # Remove the prefix to get the rest of the key
            key_parts = env_key[len(prefix):].split("__")

            if len(key_parts) < 2:
                continue  # Need at least MODEL_NAME and FIELD

            # First part is the model name (in uppercase from env var)
            model_name = key_parts[0].lower()

            # Remaining parts form the field path
            field_parts = key_parts[1:]

            # Initialize model dict if it doesn't exist
            if model_name not in models_dict:
                models_dict[model_name] = {}

            # Navigate/create nested structure
            current = models_dict[model_name]
            for i, part in enumerate(field_parts[:-1]):
                field_name = part.lower()
                if field_name not in current:
                    current[field_name] = {}
                current = current[field_name]

            # Set the final value
            final_field = field_parts[-1].lower()

            # Try to convert the value to appropriate type
            value = self._convert_env_value(env_value)
            current[final_field] = value

        # Filter for Triton backend models only
        triton_models = {}
        for model_name, model_config in models_dict.items():
            if model_config.get('backend') == 'triton':
                triton_models[model_name] = model_config

        # Create and return ModelsConfig
        config_data = {'models': triton_models}
        return ModelsConfig(**config_data)

    def _convert_env_value(self, value: str):
        """Convert environment variable string to appropriate Python type.

        Args:
            value: String value from environment variable

        Returns:
            Converted value (int, float, bool, or string)
        """
        # Try to convert to bool first (before numeric checks)
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'

        # Try to convert to int
        if value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
            return int(value)

        # Try to convert to float
        try:
            float_value = float(value)
            return float_value
        except ValueError:
            pass

        # Return as string
        return value

    def build_model_repo(self, config: ModelsConfig, repo_folder: str):
        for m in config.models:
            model_config = config.models[m]
            if model_config.type not in SUPPORTED_MODEL_TYPES:
                raise ValueError(f"Unsupported model type: {model_config.type}")

            name = m
            if model_config.triton_model_name is not None:
                name = model_config.triton_model_name
            if model_config.type == "gector":
                self.build_gector_repo(
                    name=name,
                    pretrained_model_name_or_path=model_config.pretrained_model_name_or_path,
                    repo_folder=repo_folder
                )
            elif model_config.type == "grammared_classifier":
                self.build_grammared_classifier_repo(
                    name=name,
                    pretrained_model_name_or_path=model_config.pretrained_model_name_or_path,
                    repo_folder=repo_folder
                )
            else:
                raise ValueError(f"Unsupported model type: {model_config.type}")

    def build_gector_repo(self, name:str, pretrained_model_name_or_path:str, repo_folder:str, model_version:int=1):
        output_folder = Path(repo_folder) / name
        model_folder = output_folder / str(model_version)
        config_file = self.jinja_env.get_template("gector_config.pbtxt.jinja").render({
            "model_name": name,
            "pretrained_model_name_or_path": pretrained_model_name_or_path
        })

        model_file = self.jinja_env.get_template("gector_model.py.jinja").render()
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        Path(model_folder).mkdir(parents=True, exist_ok=True)
        with open(output_folder / "config.pbtxt", "w") as f:
            f.write(config_file)

        with open(model_folder / "model.py", "w") as f:
            f.write(model_file)


    def build_grammared_classifier_repo(self, name:str, pretrained_model_name_or_path:str, repo_folder:str, model_version:int=1):
        output_folder = Path(repo_folder) / name
        model_folder = output_folder / str(model_version)
        config_file = self.jinja_env.get_template("grammared_classifier_config.pbtxt.jinja").render({
            "model_name": name,
            "pretrained_model_name_or_path": pretrained_model_name_or_path
        })

        model_file = self.jinja_env.get_template("grammared_classifier_model.py.jinja").render()
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        Path(model_folder).mkdir(parents=True, exist_ok=True)
        with open(output_folder / "config.pbtxt", "w") as f:
            f.write(config_file)

        with open(model_folder / "model.py", "w") as f:
            f.write(model_file)