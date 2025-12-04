"""
GECToR (Grammatical Error Correction: Tag, Not Rewrite) Model for Triton Inference Server.

This model uses the HuggingFace transformers library to serve the
gotutiyan/gector-roberta-base-5k model for grammar error correction.
"""

import json
import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import AutoModelForTokenClassification
import torch


class TritonPythonModel:
    """Python model for GECToR grammar error correction."""

    def initialize(self, args):
        """
        Initialize the model.
        
        Args:
            args: Dictionary containing model configuration
        """
        self.model_config = json.loads(args['model_config'])
        
        # Get model instance device configuration
        model_instance_device_id = args['model_instance_device_id']
        
        # Determine device
        if model_instance_device_id == 'CPU':
            self.device = 'cpu'
        else:
            self.device = f'cuda:{model_instance_device_id}'
        
        # Initialize attributes for cleanup
        self.model = None
        
        # Load the HuggingFace model
        model_name = "gotutiyan/gector-roberta-base-5k"
        
        try:
            self.model = AutoModelForTokenClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_name}: {str(e)}")

    def execute(self, requests):
        """
        Execute inference on a batch of requests.
        
        Args:
            requests: List of pb_utils.InferenceRequest
            
        Returns:
            List of pb_utils.InferenceResponse
        """
        responses = []
        
        for request in requests:
            try:
                # Get input tensors
                input_ids_tensor = pb_utils.get_input_tensor_by_name(request, "input_ids")
                attention_mask_tensor = pb_utils.get_input_tensor_by_name(request, "attention_mask")
                
                # Convert to numpy arrays
                input_ids_np = input_ids_tensor.as_numpy()
                attention_mask_np = attention_mask_tensor.as_numpy()
                
                # Convert to torch tensors and move to device
                input_ids = torch.from_numpy(input_ids_np).to(self.device)
                attention_mask = torch.from_numpy(attention_mask_np).to(self.device)
                
                # Run inference
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                
                # Get logits
                logits = outputs.logits
                
                # Convert logits to numpy
                logits_np = logits.cpu().numpy().astype(np.float32)
                
                # For GECToR, we return the same logits for both outputs
                # logits_labels: the main classification logits
                # logits_d: detection logits (for GECToR this is the same)
                logits_labels_np = logits_np
                logits_d_np = logits_np
                
                # Create output tensors
                output_logits_labels = pb_utils.Tensor(
                    "logits_labels",
                    logits_labels_np
                )
                
                output_logits_d = pb_utils.Tensor(
                    "logits_d",
                    logits_d_np
                )
                
                # Create inference response
                response = pb_utils.InferenceResponse(
                    output_tensors=[output_logits_labels, output_logits_d]
                )
                
            except Exception as e:
                # Return error response
                error_msg = f"Error during inference: {str(e)}"
                response = pb_utils.InferenceResponse(
                    output_tensors=[],
                    error=pb_utils.TritonError(error_msg)
                )
            
            responses.append(response)
        
        return responses

    def finalize(self):
        """Clean up resources."""
        # Clean up model safely
        if hasattr(self, 'model') and self.model is not None:
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
