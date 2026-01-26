"""
GECToR (Grammatical Error Correction: Tag, Not Rewrite) Model for Triton Inference Server.

This model uses the HuggingFace transformers library to serve the
gotutiyan/gector-bert-base-cased-5k model for grammar error correction.
"""
import logging
import json
import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import AutoModel
import torch
from gector import GECToR
from grammared_language.utils.config_parser import get_model_config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)  # Set default level to WARNING


class TritonGectorPythonModel:
    """Python model for GECToR grammar error correction."""

    def initialize(self, args):
        """
        Initialize the model.

        Args:
            args: Dictionary containing model configuration
        """
        self.model_config = json.loads(args['model_config'])
        self.grammared_language_model_config = json.loads(
            self.model_config.get('parameters', {}).get('grammared_language_model_config', {}).get('string_value', "{}")
        )
        self.grammared_language_model_config = get_model_config(self.model_config.get("name", "gector_model"), self.grammared_language_model_config)
        logger.warning(f"Loaded model config: {self.model_config}")
        logger.warning(f"Loading grammared_language_model_config: {self.grammared_language_model_config}...")
        # Get model instance device configuration

        model_device_type = args['model_instance_kind']
        model_instance_device_id = args['model_instance_device_id']

        if self.grammared_language_model_config.serving_config.device == 'cpu':
            self.device = 'cpu'
        elif self.grammared_language_model_config.serving_config.device == 'cuda':
            self.device = f'cuda:{model_instance_device_id}'
        else:
            # mimic 'auto' behavior
            if torch.cuda.is_available():
                self.device = f'cuda:{model_instance_device_id}'
            else:
                self.device = 'cpu'

        # Initialize attributes for cleanup
        self.model = None

        # Load the HuggingFace model
        if 'pretrained_model_name_or_path' in self.model_config['parameters']:
            model_name = self.model_config['parameters']['pretrained_model_name_or_path']['string_value']
        else:
            raise RuntimeError(f"Failed to get model name from configuration: {self.model_config}")

        try:
            self.model = GECToR.from_pretrained(model_name, device_map=self.device)

            logger.warning(f"Loaded model {model_name}")
            # logger.warning(f"Model config: {self.model.config}")
            # self.model.to(self.device)
            # first_encoder_layer = self.model.encoder.layer[0]
            # query_weights = first_encoder_layer.attention.self.query.weight
            # logger.warning(f"First encoder layer query weights: {query_weights}")
            # self.model.eval()


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
                gector_output = self.model.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                # Get logits
                logits_d = gector_output.logits_d
                logits_labels = gector_output.logits_labels

                # Convert logits to numpy
                logits_labels_np = logits_labels.cpu().detach().numpy().astype(np.float32)
                logits_d_np = logits_d.cpu().detach().numpy().astype(np.float32)
                # logger.warning(f"logits_labels_np: {logits_labels_np}")
                output_logits_labels = pb_utils.Tensor(
                    "logits_labels",
                    logits_labels_np
                )

                output_logits_d = pb_utils.Tensor(
                    "logits_d",
                    logits_d_np
                )

                # logger.warning(f"output_logits_d: {output_logits_d}")
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
