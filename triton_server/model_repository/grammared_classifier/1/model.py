"""
Grammared Classifier Model for Triton Inference Server.

This model uses the HuggingFace transformers library with a calibrated
text classification pipeline to classify grammar correction candidates.
"""
import logging
import json
import numpy as np
import triton_python_backend_utils as pb_utils
import torch
from grammared_language.grammared_classifier import CalibratedTextClassificationPipeline

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class TritonPythonModel:
    """Python model for Grammared Classifier."""

    def initialize(self, args):
        """
        Initialize the model.

        Args:
            args: Dictionary containing model configuration
        """
        logger.warning(f"initialize config: {args}")

        self.model_config = json.loads(args['model_config'])
        logger.info(f"Loaded model config: {self.model_config}")

        # Get model instance device configuration
        model_device_type = args['model_instance_kind']
        model_instance_device_id = args['model_instance_device_id']

        # Determine device
        if model_device_type == 'CPU':
            self.device = 'cpu'
        else:
            self.device = f'cuda:{model_instance_device_id}'

        # Initialize attributes for cleanup
        self.pipeline = None
        self.model = None
        self.tokenizer = None

        # Load the HuggingFace model
        if 'pretrained_model_name_or_path' in self.model_config['parameters']:
            model_name = self.model_config['parameters']['pretrained_model_name_or_path']['string_value']
        else:
            raise RuntimeError(f"Failed to get model name from configuration: {self.model_config}")

        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            logger.info(f"Loading model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

            # Move model to device
            self.model.to(self.device)
            self.model.eval()

            # Create calibrated pipeline
            self.pipeline = CalibratedTextClassificationPipeline(
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device
            )

            logger.info(f"Successfully loaded model {model_name} on device {self.device}")

            # Log calibration info
            if self.pipeline.calibrators:
                logger.info(f"Model has {len(self.pipeline.calibrators)} calibrators")
            else:
                logger.info("Model has no calibrators - using uncalibrated outputs")

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
                # Get input text tensor
                text_tensor = pb_utils.get_input_tensor_by_name(request, "TEXT")

                # Convert to numpy array and decode bytes to strings
                text_np = text_tensor.as_numpy()

                # Handle both single string and batch of strings
                # Triton sends strings as byte arrays
                texts = []
                for text_bytes in text_np:
                    if isinstance(text_bytes, bytes):
                        text_str = text_bytes.decode('utf-8')
                    else:
                        text_str = str(text_bytes)
                    texts.append(text_str)

                logger.debug(f"Processing {len(texts)} text(s)")

                # Tokenize the texts
                encoded = self.tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )

                # Move to device
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)

                # Run inference through the model
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )

                # Get logits
                logits = outputs.logits
                logits_np = logits.cpu().numpy()

                # Apply calibration
                calibrated_probs = self.pipeline._apply_calibration(logits_np)

                # Get label mappings
                if hasattr(self.model.config, 'id2label') and self.model.config.id2label:
                    id2label = self.model.config.id2label
                else:
                    id2label = {i: f"LABEL_{i}" for i in range(calibrated_probs.shape[1])}

                # Format outputs as label + score dictionaries
                results = []
                for probs in calibrated_probs:
                    # Get the predicted class (highest probability)
                    predicted_idx = int(np.argmax(probs))
                    predicted_label = id2label[predicted_idx]
                    predicted_score = float(probs[predicted_idx])

                    result = {
                        'label': predicted_label,
                        'score': predicted_score
                    }
                    results.append(result)

                # Convert results to JSON strings
                json_results = [json.dumps(result) for result in results]

                # Create numpy array of JSON strings
                # For single input, return one result; for batch, return list
                output_json_np = np.array(json_results, dtype=object)

                # Create output tensor
                output_json = pb_utils.Tensor(
                    "OUTPUT",
                    output_json_np
                )

                # Create inference response
                response = pb_utils.InferenceResponse(
                    output_tensors=[output_json]
                )

            except Exception as e:
                # Return error response
                error_msg = f"Error during inference: {str(e)}"
                logger.error(error_msg, exc_info=True)
                response = pb_utils.InferenceResponse(
                    output_tensors=[],
                    error=pb_utils.TritonError(error_msg)
                )

            responses.append(response)

        return responses

    def finalize(self):
        """Clean up resources."""
        # Clean up model safely
        if hasattr(self, 'pipeline') and self.pipeline is not None:
            del self.pipeline
        if hasattr(self, 'model') and self.model is not None:
            del self.model
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Model finalized and resources cleaned up")
