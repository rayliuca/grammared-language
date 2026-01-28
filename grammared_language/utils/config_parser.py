"""Utilities for parsing model configuration files."""

from pathlib import Path
from typing import Dict, Any, Optional, List, Literal, Union
import yaml
import os
from pydantic import BaseModel, ConfigDict, Field, field_validator

from grammared_language.clients.base_client import BaseClient

DEFAULT_MODEL_CONFIG_PATH = "/default_model_config.yaml"
MODEL_CONFIG_PATH = "/model_config.yaml"
MODEL_REPO_FOLDER = "/models"

import logging
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ServingConfig(BaseModel):
    """Configuration for model serving settings."""
    
    model_config = ConfigDict(extra='allow')
    
    triton_host: Optional[str] = 'localhost'
    triton_port: Optional[int] = 8001
    triton_model_name: Optional[str] = None
    triton_protocol: Literal['grpc', 'http'] = 'grpc'  # 'grpc' or 'http'
    pretrained_model_name_or_path: Optional[str] = None
    backend: Optional[str] = None
    device: Optional[str] = None
    

class ModelInitConfig(BaseModel):
    """Configuration for model inference settings."""
    
    model_config = ConfigDict(extra='allow')
    


class ModelInferenceConfig(BaseModel):
    """Configuration for model inference settings."""
    
    model_config = ConfigDict(extra='allow')
    
    # temperature: Optional[float] = None
    # max_length: Optional[int] = None
    # num_beams: Optional[int] = None


class GrammaredConfig(BaseModel):
    """Configuration for grammared-specific settings."""
    
    model_config = ConfigDict(extra='allow')
    
    prompt_template: Optional[str] = None
    error_classifier: Optional[str] = None


class BaseModelConfig(BaseModel):
    """Base configuration for all model types."""
    
    model_config = ConfigDict(extra='allow')  # Allow additional fields for model-specific configs
    
    type: str
    backend: Literal[
        'triton', 
        # 'local', # Local inference
        'openai' # OpenAI compatiable api
        ] = 'triton'
    
    # Nested config format
    serving_config: ServingConfig
    model_init_config: Optional[ModelInitConfig] = Field(default=ModelInitConfig(), alias='model_config')
    model_inference_config: Optional[ModelInferenceConfig] = ModelInferenceConfig()
    grammared_config: Optional[GrammaredConfig] = GrammaredConfig()

class GectorConfig(BaseModelConfig):
    """Configuration for GECToR models."""
    
    type: Literal['gector'] = 'gector'


class GrammaredClassifierConfig(BaseModelConfig):
    """Configuration for Grammared Classifier models."""
    
    type: Literal['grammared_classifier'] = 'grammared_classifier'


class Text2TextInferenceConfig(BaseModel):
    """Configuration for Text2Text model inference settings."""
    
    backend: Literal['transformers', 'ort'] = 'transformers'
    
    model_config = ConfigDict(extra='allow')


class Text2TextConfig(BaseModelConfig):
    """Configuration for Text2Text models."""
    
    type: Literal['text2text'] = 'text2text'
    config: Optional[Text2TextInferenceConfig] = None

class CoEditConfig(Text2TextConfig):
    """Configuration for CoEdit models."""
    
    type: Literal['coedit'] = 'coedit'


class OpenAIConfig(BaseModelConfig):
    """Configuration for OpenAI compatible models."""
    
    type: Literal['openai'] = 'openai'

    openai_base_url: Optional[str] = "http://localhost:11434/v1/"
    openai_api_key: Optional[str] = "ollama"


MODEL_CONFIG_REGISTRY = {
    'gector': GectorConfig,
    'grammared_classifier': GrammaredClassifierConfig,
    'coedit': CoEditConfig,
    'text2text': Text2TextConfig,
    'openai': OpenAIConfig,
}

def get_model_config(model_name, model_config_dict: Dict[str, Any]) -> BaseModelConfig:
    model_type = model_config_dict.get('type')
    if model_type not in MODEL_CONFIG_REGISTRY:
        raise ValueError(f"Unknown model type for {model_name}: {model_type}")
    return MODEL_CONFIG_REGISTRY[model_type](**model_config_dict)

class ModelsConfig(BaseModel):
    """Container for multiple model configurations."""
    
    models: Dict[str, Union[GectorConfig, GrammaredClassifierConfig, CoEditConfig, BaseModelConfig]]
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelsConfig':
        """Create ModelsConfig from a dictionary."""
        models = {}
        for model_name, model_config in config_dict.items():
            models[model_name] = get_model_config(model_name, model_config)
        return cls(models=models)


def load_config_from_file(config_path: str) -> ModelsConfig:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        ModelsConfig containing validated model configurations
        
    Raises:
        FileNotFoundError: If the config file doesn't exist
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config_dict = yaml.safe_load(f)
    config = ModelsConfig.from_dict(config_dict)
    logger.warning(config)
    return config


def load_config_from_env(prefix: str = "GRAMMARED_LANGUAGE") -> ModelsConfig:
    """
    Load configuration from environment variables.
    
    Environment variables should follow the pattern:
    {PREFIX}__MODELS__{MODEL_NAME}__{KEY}={VALUE}
    
    Example:
        GRAMMARED_LANGUAGE__MODELS__GECTOR_DEBERTA_LARGE__TYPE=gector
        GRAMMARED_LANGUAGE__MODELS__GECTOR_DEBERTA_LARGE__BACKEND=triton
        GRAMMARED_LANGUAGE__MODELS__GECTOR_DEBERTA_LARGE__SERVING_CONFIG__TRITON_HOSTNAME=localhost
    
    Args:
        prefix: Environment variable prefix (default: "GRAMMARED_LANGUAGE")
        
    Returns:
        ModelsConfig containing validated model configurations
        
    Raises:
        ValueError: If no matching environment variables are found
    """
    prefix_with_sep = f"{prefix}__"
    config_dict: Dict[str, Any] = {}
    
    # Find all environment variables with the prefix
    env_vars = {k: v for k, v in os.environ.items() if k.startswith(prefix_with_sep)}
    
    if not env_vars:
        raise ValueError(f"No environment variables found with prefix: {prefix_with_sep}")
    
    for env_key, env_value in env_vars.items():
        # Remove prefix and split by double underscore
        key_parts = env_key[len(prefix_with_sep):].split("__")
        
        # Convert keys to lowercase for model names
        key_parts = [part.lower() for part in key_parts]
        
        # Build nested dictionary
        current = config_dict
        for i, part in enumerate(key_parts[:-1]):
            if part not in current:
                current[part] = {}
            current = current[part]
        
        # Set the final value, attempting type conversion
        final_key = key_parts[-1]
        
        # Try to convert value to appropriate type
        try:
            # Try boolean
            if env_value.lower() in ('true', 'false'):
                current[final_key] = env_value.lower() == 'true'
            # Try integer
            elif env_value.isdigit():
                current[final_key] = int(env_value)
            # Try float
            elif '.' in env_value and env_value.replace('.', '').replace('-', '').isdigit():
                current[final_key] = float(env_value)
            # Keep as string
            else:
                current[final_key] = env_value
        except (ValueError, AttributeError):
            current[final_key] = env_value
    
    # Extract models dict (skip the "models" key level)
    models_dict = config_dict.get("models", {})
    
    if not models_dict:
        raise ValueError(f"No models configuration found in environment variables")
    
    logger.info(f"Loaded configuration for {len(models_dict)} model(s) from environment variables")
    
    return ModelsConfig.from_dict(models_dict)


def create_client_from_config(
    model_name: str, 
    config: Union[GectorConfig, GrammaredClassifierConfig, CoEditConfig, Dict[str, Any]]
) -> Optional[BaseClient]:
    """
    Create a client instance from configuration.
    
    Args:
        model_name: Name of the model
        config: Configuration (Pydantic model or dictionary)
        
    Returns:
        Initialized client instance or None if creation fails
    """
    # Convert dict to Pydantic model if needed
    if isinstance(config, dict):
        model_type = config.get('type')
        if model_type not in MODEL_CONFIG_REGISTRY:
            raise ValueError(f"Unknown model type for {model_name}: {model_type}")
        config = MODEL_CONFIG_REGISTRY[model_type](**config)
    
    try:
        # Import appropriate client class based on type
        if isinstance(config, GectorConfig):
            from grammared_language.clients.gector_client import GectorClient
            
            # GectorClient expects model_id and triton_model_name
            client_params = vars(config.serving_config)
            client_params.update(vars(config.grammared_config) if config.grammared_config else {})
            
            return GectorClient(**client_params)
        
        elif isinstance(config, GrammaredClassifierConfig):
            from grammared_language.clients.grammar_classification_client import GrammarClassificationClient
                        
            # Filter out non-client parameters
            client_params = vars(config.serving_config)
            client_params.update(vars(config.grammared_config) if config.grammared_config else {})

            logger.warning(f"Creating GrammarClassificationClient with params: {client_params}")
            return GrammarClassificationClient(**client_params)
            
        elif isinstance(config, CoEditConfig):
            from grammared_language.clients.coedit_client import CoEditClient
            
            client_params = vars(config.serving_config)
            client_params.update(vars(config.grammared_config) if config.grammared_config else {})
            
            logger.warning(f"Creating CoEditClient with params: {client_params}")
            return CoEditClient(**client_params)
            
    except Exception as e:
        logger.error(f"Failed to create client for {model_name}: {e}")
    
    return None


"""
1. check if MODEL_REPO_FOLDER exists
2. check MODEL_CONFIG_PATH
3. check environment variable
4. use DEFAULT_MODEL_CONFIG_PATH
"""
def get_config(config_path:str=MODEL_CONFIG_PATH, use_env: bool=True, backup_config_path=DEFAULT_MODEL_CONFIG_PATH) -> ModelsConfig:
    env_vars = {k: v for k, v in os.environ.items() if k.startswith("GRAMMARED_LANGUAGE__")}
    # Determine which config path to use
    if os.path.isfile(config_path):
        logger.info(f"Loading model configuration from file: {config_path}")
        config = load_config_from_file(config_path)
    elif env_vars and use_env:
        logger.info(f"Loading model configuration from environment variables")
        config = load_config_from_env()
    else:
        logger.info(f"No config path provided. Loading model configuration from backup file: {backup_config_path}")
        config = load_config_from_file(backup_config_path)
    return config


def create_clients_from_config(config_path: str=None, use_env: bool=True, backup_config_path: str="/default_model_config.yaml") -> List[BaseClient]:
    """
    Initialize multiple clients from a configuration file or environment variables.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        List of initialized client instances
    """
    models_config = get_config(config_path=config_path, use_env=use_env, backup_config_path=backup_config_path)
    clients = []
    
    for m_name, m_config in models_config.models.items():
        client = create_client_from_config(m_name, m_config)
        if client:
            clients.append(client)
    
    return clients
