import numpy as np
from transformers import TextClassificationPipeline
from typing import Dict, List, Union, Optional
import torch


def load_logistic_weights(weight_dict, lr_model=None):
    """Load weights & bias into a sklearn LogisticRegression from a dict."""
    if lr_model is None:
        from sklearn.linear_model import LogisticRegression
        lr_model = LogisticRegression()
    lr_model.coef_ = np.array(weight_dict["coef"])
    lr_model.intercept_ = np.array(weight_dict["intercept"])
    lr_model.classes_ = np.array(weight_dict["classes"])
    return lr_model


class CalibratedTextClassificationPipeline(TextClassificationPipeline):
    """
    Text classification pipeline with probability calibration support.

    Loads calibrator weights from model. config. probability_calibrator_weights
    and applies calibration to the model outputs.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calibrators = None
        self._load_calibrators()

    def _load_calibrators(self):
        """Load calibrators from model config if available."""
        if hasattr(self.model.config, 'probability_calibrator_weights'):
            calibrator_weights = self.model.config.probability_calibrator_weights
            if calibrator_weights:
                self.calibrators = [
                    load_logistic_weights(weights)
                    for weights in calibrator_weights
                ]
                print(f"Loaded {len(self.calibrators)} calibrators from model config")
        else:
            print("No calibrators found in model config - using uncalibrated outputs")

    def _apply_calibration(self, logits: np.ndarray) -> np.ndarray:
        """
        Apply calibration to logits.

        Args:
            logits: Raw model logits, shape (batch_size, num_classes)

        Returns:
            Calibrated probabilities, shape (batch_size, num_classes)
        """
        if self.calibrators is None:
            # Fallback to softmax if no calibrators
            from scipy.special import softmax
            return softmax(logits, axis=1)

        # Apply each calibrator to get calibrated probability for each class
        calibrated_probs = np.column_stack([
            calibrator.predict_proba(logits)[:, 1]  # Probability for positive class
            for calibrator in self.calibrators
        ])

        # Normalize to sum to 1
        calibrated_probs = calibrated_probs / calibrated_probs.sum(axis=1, keepdims=True)

        return calibrated_probs

    def postprocess(self, model_outputs, function_to_apply=None, top_k=1, **kwargs):
        """
        Postprocess model outputs with calibration.

        Args:
            model_outputs: Raw model outputs containing logits
            function_to_apply: Optional function to apply (overridden by calibration)
            top_k: Number of top predictions to return

        Returns:
            List of dictionaries containing labels and calibrated scores
        """
        # Extract logits
        if isinstance(model_outputs, dict):
            logits = model_outputs["logits"]
        else:
            logits = model_outputs[0]

        # Convert to numpy
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().cpu().numpy()

        # Apply calibration
        calibrated_probs = self._apply_calibration(logits)

        # Get label mappings
        if self.model.config.id2label:
            id2label = self.model.config.id2label
        else:
            id2label = {i: f"LABEL_{i}" for i in range(calibrated_probs.shape[1])}

        # Format outputs
        outputs = []
        for probs in calibrated_probs:
            # Get top-k predictions
            top_indices = np.argsort(probs)[::-1][:top_k]

            result = [
                {
                    "label": id2label[int(idx)],
                    "score": float(probs[idx])
                }
                for idx in top_indices
            ]

            outputs.append(result if top_k > 1 else result[0])

        return outputs[0]

    def predict_proba(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Get calibrated probability predictions for texts.

        Args:
            texts: Single text or list of texts

        Returns:
            Calibrated probabilities, shape (num_texts, num_classes)
        """
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False

        # Get model outputs
        outputs = self(texts, top_k=self.model.config.num_labels)

        # Extract probabilities
        if single_input:
            outputs = [outputs]

        probs = []
        for output in outputs:
            if single_input:
                output = output[0]
            sorted_output = sorted(output,
                                   key=lambda x: self.model.config.label2id[x['label']])
            probs.append([item['score'] for item in sorted_output])

        probs = np.array(probs)

        return probs[0] if single_input else probs

    def predict(self, texts: Union[str, List[str]], return_dict=True) -> Union[str, List[dict], dict]:
        """
        Get calibrated predicted labels for texts.

        Args:
            texts: Single text or list of texts
            return_dict: return dict with score or not

        Returns:
            Predicted label(s)
        """
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False

        # Get predictions
        outputs = self(texts, top_k=1)

        if not return_dict:
            outputs = [o['label'] for o in outputs]

        # Extract labels
        if single_input:
            return outputs[0]
        else:
            return outputs

    def get_uncalibrated_proba(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Get uncalibrated probability predictions (raw softmax).

        Args:
            texts: Single text or list of texts

        Returns:
            Uncalibrated probabilities, shape (num_texts, num_classes)
        """
        # Temporarily disable calibrators
        calibrators_backup = self.calibrators
        self.calibrators = None

        try:
            probs = self.predict_proba(texts)
        finally:
            # Restore calibrators
            self.calibrators = calibrators_backup

        return probs

    @classmethod
    # Convenience function to create the pipeline
    def from_pretrained(
            cls,
            model_name_or_path: str,
            **kwargs
    ):
        """
        Create a calibrated text classification pipeline.

        Args:
            model_name_or_path: HuggingFace model identifier or local path
            **kwargs:  Additional arguments passed to pipeline

        Returns:
            CalibratedTextClassificationPipeline instance

        Example:
            >>> pipeline = CalibratedTextClassificationPipeline.from_pretrained("rayliuca/grammared-classifier-deberta-v3-base")
            >>> predictions = pipeline.predict(["This is great!", "This is bad!"])
            >>> probs = pipeline.predict_proba(["This is great! "])
        """
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)

        return cls(
            model=model,
            tokenizer=tokenizer,
            **kwargs
        )

# class GrammarClassifierModel:
#     def __init__(self, model_name: str, tokenizer=None):
#         self.pipeline = CalibratedTextClassificationPipeline.from_pretrained(model_name)
#
#         if tokenizer is None:
#             from transformers import AutoTokenizer
#             tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.tokenizer = tokenizer
#
#     def predict(self, sentence_input: str|list[str]) -> list:
#         if isinstance(sentence_input, str):
#             sentence_input = [sentence_input]
#         return self.pipeline.predict(sentence_input, top_k=1)