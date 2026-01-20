from typing import Dict, Any, Optional, Union, Literal
from pydantic import BaseModel, Field

class ModelConfig(BaseModel):
    """Configuration for an individual model.

    Flexible configuration that supports various model types and backends.
    All fields are optional or have sensible defaults to maximize flexibility.
    """
    # Core fields - flexible typing
    type: Optional[str] = Literal["gector", "coedit", "deliterater", "grammared_classifier", "llm"]
    backend: Optional[str] = Literal["triton", "local"]

    # Optional fields
    pretrained_model_name_or_path: Optional[str] = Field(
        None, description="HuggingFace model name or local path to pretrained model"
    )
    triton_model_name: Optional[str] = Field(
        None, description="Name of the model in Triton server"
    )
    triton_hostname: Optional[str] = Field(
        None, description="Hostname of the Triton server"
    )
    triton_port: Optional[int] = Field(None, description="Port of the Triton server")

    # Flexible configuration dictionaries
    model_kwargs: Optional[Dict[str, Any]] = Field(
        None, description="Additional model-specific arguments (e.g., temperature, max_new_tokens)"
    )
    configs: Optional[Any] = Field(
        None, description="Model-specific configurations"
    )
    rule_generation: Optional[Any] = Field(
        None, description="Rule generation configuration (if applicable)"
    )

    class Config:
        extra = "allow"  # Allow any additional fields


class ModelsConfig(BaseModel):
    """Top-level configuration containing all models.

    Flexible structure that allows for various model configurations.
    """
    models: Dict[str, Union[ModelConfig, Dict[str, Any]]] = Field(
        default_factory=dict, description="Dictionary of model configurations keyed by model name"
    )

    class Config:
        extra = "allow"  # Allow additional top-level fields

# Convenience alias for the main config
Config = ModelsConfig

