from .base_client import BaseClient
from typing import Optional
import numpy as np
import jinja2

try:
    import tritonclient.http as httpclient
    import tritonclient.grpc as grpcclient
    _TRITON_HTTP_AVAILABLE = True
    _TRITON_GRPC_AVAILABLE = True
except Exception:
    try:
        import tritonclient.grpc as grpcclient
        httpclient = None
        _TRITON_HTTP_AVAILABLE = False
        _TRITON_GRPC_AVAILABLE = True
    except Exception:
        httpclient = None
        grpcclient = None
        _TRITON_HTTP_AVAILABLE = False
        _TRITON_GRPC_AVAILABLE = False

_TRITON_AVAILABLE = _TRITON_HTTP_AVAILABLE or _TRITON_GRPC_AVAILABLE


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
        triton_port: Triton server port (default: 8001 for gRPC, 8000 for HTTP)
        triton_model_version: Model version (default: "1")
        triton_protocol: Communication protocol - "grpc" or "http" (default: "grpc")
        input_name: Name of the input tensor (default: "text_input")
        output_name: Name of the output tensor (default: "text_output")
        prompt_template: Optional template string for formatting input (Jinja2-compatible).
                      Use {{ text }} as placeholder. Example: "Fix grammar: {{ text }}"
                      For chat-style models, use format like: "<|user|>\n{{ text }}<|assistant|>\n"
    """
    
    def __init__(
        self,
        model_name: str,
        *,
        triton_host: str = "localhost",
        triton_port: int = 8001,  # Default to gRPC port
        triton_model_version: str = "1",
        triton_protocol: str = "grpc",  # "grpc" or "http"
        input_name: str = "text_input",
        output_name: str = "text_output",
        prompt_template: Optional[str] = None,
        **kwargs
    ):
        if "rule_id" not in kwargs:
            kwargs["rule_id"] = model_name
        super().__init__(**kwargs)
        
        if not _TRITON_AVAILABLE:
            raise ImportError(
                "tritonclient is required for Text2TextBaseClient. "
                "Install with: pip install tritonclient[all]"
            )
        
        self.model_name = model_name
        self.triton_model_version = triton_model_version
        self.triton_protocol = triton_protocol.lower()
        self.input_name = input_name
        self.output_name = output_name
        self.prompt_template = prompt_template
        self._template = jinja2.Template(prompt_template) if prompt_template else None
        
        # Initialize Triton client based on protocol
        triton_url = f"{triton_host}:{triton_port}"
        if self.triton_protocol == "grpc":
            if not _TRITON_GRPC_AVAILABLE:
                raise ImportError(
                    "tritonclient.grpc is required for gRPC protocol. "
                    "Install with: pip install tritonclient[grpc]"
                )
            self._triton_client = grpcclient.InferenceServerClient(url=triton_url)
        else:
            if not _TRITON_HTTP_AVAILABLE:
                raise ImportError(
                    "tritonclient.http is required for HTTP protocol. "
                    "Install with: pip install tritonclient[http]"
                )
            self._triton_client = httpclient.InferenceServerClient(url=triton_url)
        
    def _preprocess(self, text: str) -> str:
        """Apply prompt template if configured."""
        if self._template:
            return self._template.render(text=text)
        return text
    
    def _predict(self, text: str|list[str]) -> str|list[str]:
        """
        Send text to Triton model and get generated output.
        
        Args:
            text: Input text (already preprocessed)
            
        Returns:
            Generated/corrected text from the model
        """
        # Prepare input as numpy array with shape [1, 1] for batching
        # First dimension is batch size, second is the string dimension
        single = True
        if isinstance(text, list):
            text_np = np.array([text], dtype=object)
            single = False
        else:
            text_np = np.array([[text]], dtype=object)
        
        # Create Triton input/output tensors based on protocol
        if self.triton_protocol == "grpc":
            inputs = [
                grpcclient.InferInput(
                    self.input_name, 
                    list(text_np.shape), 
                    "BYTES"
                )
            ]
            inputs[0].set_data_from_numpy(text_np)
            outputs = [grpcclient.InferRequestedOutput(self.output_name)]
        else:
            inputs = [
                httpclient.InferInput(
                    self.input_name, 
                    list(text_np.shape), 
                    "BYTES"
                )
            ]
            inputs[0].set_data_from_numpy(text_np)
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
        
        # Extract from batch: output_data has shape [batch_size, 1]
        def decode_result(result):
            if isinstance(result, bytes):
                return result.decode("utf-8")
            elif isinstance(result, np.ndarray):
                if result.dtype.type is np.bytes_:
                    return result.tobytes().decode("utf-8")
                return str(result)
            else:
                return str(result)
        results = [decode_result(output_data[i][0]) for i in range(len(output_data))]
        if single:
            return results[0]
        return results