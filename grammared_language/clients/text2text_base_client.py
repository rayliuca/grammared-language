from .base_client import BaseClient
from typing import Optional
import numpy as np

try:
    import tritonclient.http as httpclient
    _TRITON_AVAILABLE = True
except Exception:
    httpclient = None
    _TRITON_AVAILABLE = False


class Text2TextBaseClient(BaseClient):
    """
    Client for text-to-text generation models on Triton Inference Server.
    
    This client provides a common interface for LLM/text2text models that:
    - Accept a text input string
    - Generate corrected/transformed text output
    - Can be used for grammar correction, text editing, etc.
    
    Example models: CoEdit, T5, FLAN-T5, etc.
    
    Args:
        model_name: Name of the Triton model to use
        triton_host: Triton server host (default: "localhost")
        triton_port: Triton server port (default: 8000)
        triton_model_version: Model version (default: "1")
        input_name: Name of the input tensor (default: "text_input")
        output_name: Name of the output tensor (default: "text_output")
        prompt_template: Optional template string for formatting input. Use {text} as placeholder.
                        Example: "Fix grammar: {text}"
    """
    
    def __init__(
        self,
        model_name: str,
        *,
        triton_host: str = "localhost",
        triton_port: int = 8000,
        triton_model_version: str = "1",
        input_name: str = "text_input",
        output_name: str = "text_output",
        prompt_template: Optional[str] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        if not _TRITON_AVAILABLE:
            raise ImportError(
                "tritonclient is required for Text2TextBaseClient. "
                "Install with: pip install tritonclient[all]"
            )
        
        self.model_name = model_name
        self.triton_model_version = triton_model_version
        self.input_name = input_name
        self.output_name = output_name
        self.prompt_template = prompt_template
        
        # Initialize Triton HTTP client
        triton_url = f"{triton_host}:{triton_port}"
        self._triton_client = httpclient.InferenceServerClient(url=triton_url)
        
    def _preprocess(self, text: str) -> str:
        """Apply prompt template if configured."""
        if self.prompt_template:
            return self.prompt_template.format(text=text)
        return text
    
    def _predict(self, text: str) -> str:
        """
        Send text to Triton model and get generated output.
        
        Args:
            text: Input text (already preprocessed)
            
        Returns:
            Generated/corrected text from the model
        """
        # Prepare input as numpy array with shape [1]
        text_np = np.array([text], dtype=object)
        
        # Create Triton input tensor
        inputs = [
            httpclient.InferInput(
                self.input_name, 
                list(text_np.shape), 
                "BYTES"
            )
        ]
        inputs[0].set_data_from_numpy(text_np)
        
        # Create Triton output request
        outputs = [httpclient.InferRequestedOutput(self.output_name)]
        
        # Send inference request
        response = self._triton_client.infer(
            model_name=self.model_name,
            model_version=self.triton_model_version,
            inputs=inputs,
            outputs=outputs,
        )
        
        # Extract output text
        output_data = response.as_numpy(self.output_name)
        
        # Handle different output formats
        if len(output_data) == 0:
            return text  # Return original if no output
        
        result = output_data[0]
        if isinstance(result, bytes):
            return result.decode("utf-8")
        elif isinstance(result, np.ndarray):
            if result.dtype.type is np.bytes_:
                return result.tobytes().decode("utf-8")
            return str(result)
        else:
            return str(result)