"""
Grammared Classifier Model for Triton Inference Server.

This model uses the HuggingFace transformers library with a calibrated
text classification pipeline to classify grammar correction candidates.
"""

from grammared_language.triton.triton_grammared_classifier_model import TritonGrammaredClassifierPythonModel

class TritonPythonModel(TritonGrammaredClassifierPythonModel):
    ...