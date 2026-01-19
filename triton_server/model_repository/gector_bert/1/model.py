"""
GECToR (Grammatical Error Correction: Tag, Not Rewrite) Model for Triton Inference Server.

This model uses the HuggingFace transformers library to serve the
gotutiyan/gector-bert-base-cased-5k model for grammar error correction.
"""
from grammared_language.triton.triton_gector_model import TritonGectorPythonModel

class TritonPythonModel(TritonGectorPythonModel):
    ...