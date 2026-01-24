from .text2text_base_client import Text2TextBaseClient
from typing import Optional


class CoEditClient(Text2TextBaseClient):
    """
    Client for CoEdit grammar correction models on Triton Inference Server.
    
    CoEdit is a text editing model trained for grammar correction and text refinement.
    This client provides a convenient interface with sensible defaults for CoEdit models.
    
    Model: grammarly/coedit-large, grammarly/coedit-xl
    Paper: https://arxiv.org/abs/2305.09857
    
    Args:
        model_name: Name of the Triton model (default: "coedit_large")
        triton_host: Triton server host (default: "localhost")
        triton_port: Triton server port (default: 8001 for gRPC, 8000 for HTTP)
        triton_model_version: Model version (default: "1")
        triton_protocol: Communication protocol - "grpc" or "http" (default: "grpc")
        prompt_template: Optional template string for formatting input (Jinja2-compatible).
                      Use {{ text }} as placeholder. Defaults to "Fix grammatical errors: {{ text }}".
                      Example: "Fix grammar: {{ text }}"
                      For chat-style: "<|user|>\n{{ text }}<|assistant|>\n"
        **kwargs: Additional arguments passed to Text2TextBaseClient
    
    Example:
        >>> client = CoEditClient()
        >>> result = client.predict("She go to the store yesterday.")
        >>> print(result.matches)
    """

    DEFAULT_PROMPT_TEMPLATE = "Fix grammatical errors: {{ text }}"
    
    def __init__(
        self,
        model_name: str = "coedit_large",
        *,
        triton_host: str = "localhost",
        triton_port: int = 8001,  # Default to gRPC port
        triton_model_version: str = "1",
        triton_protocol: str = "grpc",  # "grpc" or "http"
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        **kwargs
    ):
        # Initialize parent with CoEdit-specific defaults
        super().__init__(
            model_name=model_name,
            triton_host=triton_host,
            triton_port=triton_port,
            triton_model_version=triton_model_version,
            triton_protocol=triton_protocol,
            input_name="text_input",
            output_name="text_output",
            prompt_template=prompt_template,
            **kwargs
        )
