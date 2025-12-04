"""
GECToR (Grammatical Error Correction: Tag, Not Rewrite) Model for Triton Inference Server.

This model uses the HuggingFace transformers library to serve the
gotutiyan/gector-roberta-base-5k model for grammar error correction.
"""

import json
import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer, AutoModelForTokenClassification
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
        
        # Get model instance configuration
        model_instance_name = args['model_instance_name']
        model_instance_device_id = args['model_instance_device_id']
        
        # Determine device
        if model_instance_device_id == 'CPU':
            self.device = 'cpu'
        else:
            self.device = f'cuda:{model_instance_device_id}'
        
        # Load the HuggingFace model and tokenizer
        model_name = "gotutiyan/gector-roberta-base-5k"
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForTokenClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # Store label information
            self.id2label = self.model.config.id2label
            self.label2id = self.model.config.label2id
            
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
            # Get input text
            input_text = pb_utils.get_input_tensor_by_name(request, "INPUT_TEXT")
            input_text_data = input_text.as_numpy()
            
            # Decode text from bytes if necessary
            if input_text_data.dtype == np.object_:
                texts = [text.decode('utf-8') if isinstance(text, bytes) else text 
                        for text in input_text_data.flatten()]
            else:
                texts = input_text_data.flatten().tolist()
            
            try:
                # Process each text
                all_corrections = []
                all_labels = []
                all_confidences = []
                
                for text in texts:
                    # Tokenize input
                    inputs = self.tokenizer(
                        text,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512
                    )
                    
                    # Move to device
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    # Run inference
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                    
                    # Get predictions
                    logits = outputs.logits
                    predictions = torch.argmax(logits, dim=-1)
                    probabilities = torch.softmax(logits, dim=-1)
                    
                    # Convert predictions to labels
                    predicted_labels = [
                        self.id2label[pred.item()] 
                        for pred in predictions[0]
                    ]
                    
                    # Get confidence scores
                    confidence_scores = [
                        prob[pred].item() 
                        for prob, pred in zip(probabilities[0], predictions[0])
                    ]
                    
                    # Get tokens for reference
                    tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
                    
                    # Format corrections
                    corrections = self._format_corrections(
                        text, tokens, predicted_labels, confidence_scores
                    )
                    
                    all_corrections.append(json.dumps(corrections))
                    all_labels.append(json.dumps(predicted_labels))
                    all_confidences.append(json.dumps(confidence_scores))
                
                # Create output tensors
                output_corrections = pb_utils.Tensor(
                    "CORRECTIONS",
                    np.array(all_corrections, dtype=np.object_)
                )
                
                output_labels = pb_utils.Tensor(
                    "LABELS",
                    np.array(all_labels, dtype=np.object_)
                )
                
                output_confidences = pb_utils.Tensor(
                    "CONFIDENCES",
                    np.array(all_confidences, dtype=np.object_)
                )
                
                # Create inference response
                response = pb_utils.InferenceResponse(
                    output_tensors=[output_corrections, output_labels, output_confidences]
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

    def _format_corrections(self, text, tokens, labels, confidences):
        """
        Format the corrections from model predictions.
        
        Args:
            text: Original input text
            tokens: Tokenized words
            labels: Predicted labels for each token
            confidences: Confidence scores for each prediction
            
        Returns:
            List of correction dictionaries
        """
        corrections = []
        
        # Skip special tokens and process actual word tokens
        for i, (token, label, confidence) in enumerate(zip(tokens, labels, confidences)):
            # Skip special tokens like [CLS], [SEP], [PAD]
            if token in ['<s>', '</s>', '<pad>', '[CLS]', '[SEP]', '[PAD]']:
                continue
            
            # If label is not KEEP (indicating a correction is needed)
            if label != 'KEEP' and label != '$KEEP':
                correction = {
                    'token': token,
                    'label': label,
                    'confidence': float(confidence),
                    'position': i
                }
                corrections.append(correction)
        
        return corrections

    def finalize(self):
        """Clean up resources."""
        # Clean up model and tokenizer
        del self.model
        del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
