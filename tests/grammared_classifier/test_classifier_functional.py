import pytest
import os
import torch
from transformers import AutoTokenizer
from grammared_language.grammared_classifier.grammared_classifier_model import TritonGrammaredClassifierModel
from grammared_language.grammared_classifier.classifier_pipeline import CalibratedTextClassificationPipeline

# Use a small model for testing to minimize download time
TEST_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

# Check if we should run non-hermetic tests
RUN_NON_HERMETIC = os.getenv("RUN_NON_HERMETIC", "false").lower() in ("true", "1", "yes")

pytestmark = pytest.mark.skipif(
    not RUN_NON_HERMETIC,
    reason="RUN_NON_HERMETIC=true is not set"
)

@pytest.fixture(scope="module")
def real_classifier_model():
    """Initialize a real model and pipeline for functional testing."""
    # We use a standard HF model to test the pipeline and model wrapper
    # in a realistic scenario.
    model = TritonGrammaredClassifierModel(model_name=TEST_MODEL_NAME)
    return model

@pytest.mark.functional
class TestGrammaredClassifierFunctional:
    """Functional tests for Grammared Classifier that download and run real models."""

    def test_predict_real_output(self, real_classifier_model):
        """Test prediction with a real model."""
        sentence = "This is a great movie!"
        results = real_classifier_model.predict(sentence)
        
        # Results should be a list containing one prediction (top_k=1)
        # For SST-2, it should return POSITIVE or NEGATIVE
        assert isinstance(results, dict)
        assert "label" in results
        assert "score" in results
        assert isinstance(results["score"], float)
        assert 0.0 <= results["score"] <= 1.0

    def test_predict_batch_real_output(self, real_classifier_model):
        """Test batch prediction with a real model."""
        sentences = ["I love this!", "I hate this."]
        results = real_classifier_model.predict(sentences)
        
        assert isinstance(results, list)
        assert len(results) == 2
        for res in results:
            assert isinstance(res, dict)
            assert "label" in res
            assert 0.0 <= res["score"] <= 1.0

    def test_pipeline_instantiation(self):
        """Test that our custom pipeline can be instantiated via HF factory."""
        from transformers import pipeline
        
        # Note: In real usage, the model config would have the calibrator weights.
        # Here we just verify the class can be initialized.
        classifier = CalibratedTextClassificationPipeline(
            model=TEST_MODEL_NAME,
            tokenizer=TEST_MODEL_NAME,
            task="text-classification"
        )
        
        assert isinstance(classifier, CalibratedTextClassificationPipeline)
        
        # Verify it can run
        out = classifier("Hello world")
        assert isinstance(out, list)
        assert "label" in out[0]
