"""Utilities for parsing model configuration files."""

from pathlib import Path
from typing import Dict, Any, Optional, List, Literal, Union
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from grammared_language.clients.base_client import BaseClient


class BaseModelConfig(BaseModel):
    """Base configuration for all model types."""
    
    model_config = ConfigDict(extra='allow')  # Allow additional fields for model-specific configs
    
    # type: Literal['gector', 'grammared_classifier', 'coedit']
    type: str
    backend: Literal[
        'triton', 
        # 'local', # Local inference
        'openai' # OpenAI compatiable api
        ] = 'triton'
    pretrained_model_name_or_path: Optional[str] = None
    triton_model_name: Optional[str] = None
    triton_hostname: Optional[str] = 'localhost'
    triton_port: Optional[int] = 8001
    triton_protocol: Optional[Literal[
        'grpc', 
        # 'http'
        ]] = 'grpc'

    config: Optional[Dict[str, Any]] = None
    model_kwargs: Optional[Dict[str, Any]] = None
    error_classifier: Optional[str] = None # the name of another model


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


class ModelsConfig(BaseModel):
    """Container for multiple model configurations."""
    
    models: Dict[str, Union[GectorConfig, GrammaredClassifierConfig, CoEditConfig]]
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelsConfig':
        """Create ModelsConfig from a dictionary."""
        models = {}
        for model_name, model_config in config_dict.items():
            if isinstance(model_config, dict):
                model_type = model_config.get('type')
                if model_type not in MODEL_CONFIG_REGISTRY:
                    raise ValueError(f"Unknown model type for {model_name}: {model_type}")
                models[model_name] = MODEL_CONFIG_REGISTRY[model_type](**model_config)
        return cls(models=models)


def load_config(config_path: str) -> ModelsConfig:
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
    
    return ModelsConfig.from_dict(config_dict)


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
            
            triton_model_name = config.triton_model_name if config.backend == 'triton' else None
            
            input_fields = {
                k: v for k, v in config.model_dump().items() 
                if k not in ['type']
            }
            
            return GectorClient(**input_fields)
        
        elif isinstance(config, GrammaredClassifierConfig):
            from grammared_language.clients.grammar_classification_client import GrammarClassificationClient
            
            input_fields = {
                k: v for k, v in config.model_dump().items() 
                if k not in ['type']
            }
            
            return GrammarClassificationClient(**input_fields)
            
        elif isinstance(config, CoEditConfig):
            from grammared_language.clients.coedit_client import CoEditClient
            
            triton_model_name = config.triton_model_name if config.backend == 'triton' else None
            
            input_fields = {
                k: v for k, v in config.model_dump().items() 
                if k not in ['type']
            }
            
            return CoEditClient(**input_fields)
            
    except Exception as e:
        print(f"Failed to create client for {model_name}: {e}")
    
    return None


def create_clients_from_config(config_path: str) -> List[BaseClient]:
    """
    Initialize multiple clients from a configuration file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        List of initialized client instances
    """
    models_config = load_config(config_path)
    clients = []
    
    for model_name, model_config in models_config.models.items():
        client = create_client_from_config(model_name, model_config)
        if client:
            clients.append(client)
    
    return clients
