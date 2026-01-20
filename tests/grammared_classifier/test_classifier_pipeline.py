import pytest
import numpy as np
import torch
from unittest.mock import MagicMock, patch
from grammared_language.grammared_classifier.classifier_pipeline import (
    CalibratedTextClassificationPipeline,
    load_logistic_weights
)

def test_load_logistic_weights():
    weight_dict = {
        "coef": [[1.5, -0.5]],
        "intercept": [0.1],
        "classes": [0, 1]
    }
    
    # Test loading into a new model
    lr_model = load_logistic_weights(weight_dict)
    
    from sklearn.linear_model import LogisticRegression
    assert isinstance(lr_model, LogisticRegression)
    np.testing.assert_array_equal(lr_model.coef_, np.array(weight_dict["coef"]))
    np.testing.assert_array_equal(lr_model.intercept_, np.array(weight_dict["intercept"]))
    np.testing.assert_array_equal(lr_model.classes_, np.array(weight_dict["classes"]))

def test_pipeline_initialization_no_calibrators():
    mock_model = MagicMock()
    # Mock config without calibrator weights
    mock_model.config = MagicMock()
    del mock_model.config.probability_calibrator_weights
    
    mock_tokenizer = MagicMock()
    
    with patch("transformers.TextClassificationPipeline.__init__", return_value=None):
        pipeline = CalibratedTextClassificationPipeline.__new__(CalibratedTextClassificationPipeline)
        pipeline.model = mock_model
        pipeline.calibrators = None
        pipeline._load_calibrators()
        assert pipeline.calibrators is None

def test_pipeline_initialization_with_calibrators():
    mock_model = MagicMock()
    calibrator_weights = [
        {"coef": [[1.0]], "intercept": [0.0], "classes": [0, 1]},
        {"coef": [[2.0]], "intercept": [0.1], "classes": [0, 1]}
    ]
    mock_model.config.probability_calibrator_weights = calibrator_weights
    
    mock_tokenizer = MagicMock()
    
    with patch("transformers.TextClassificationPipeline.__init__", return_value=None):
        pipeline = CalibratedTextClassificationPipeline.__new__(CalibratedTextClassificationPipeline)
        pipeline.model = mock_model
        pipeline._load_calibrators()
        assert pipeline.calibrators is not None
        assert len(pipeline.calibrators) == 2
        assert isinstance(pipeline.calibrators[0].coef_, np.ndarray)

def test_postprocess_without_calibrators():
    mock_model = MagicMock()
    del mock_model.config.probability_calibrator_weights
    mock_model.config.id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    
    mock_tokenizer = MagicMock()
    
    with patch("transformers.TextClassificationPipeline.__init__", return_value=None):
        pipeline = CalibratedTextClassificationPipeline.__new__(CalibratedTextClassificationPipeline)
        pipeline.model = mock_model
        pipeline.calibrators = None
        
        # Mock model output (logits)
        # For 2 classes, softmax of [0, 1] is approx [0.269, 0.731]
        model_outputs = {"logits": torch.tensor([[0.0, 1.0]])}
        
        result = pipeline.postprocess(model_outputs, top_k=1)
        
        # When top_k=1, it returns the dict directly, not a list of dicts
        assert result["label"] == "POSITIVE"
        assert pytest.approx(result["score"], 0.001) == 0.731

def test_postprocess_with_calibrators():
    mock_model = MagicMock()
    # Simple calibrators that just return probability for class 1
    # We mock the calibrator object directly for simplicity
    mock_calibrator1 = MagicMock()
    mock_calibrator1.predict_proba.return_value = np.array([[0.2, 0.8]])
    mock_calibrator2 = MagicMock()
    mock_calibrator2.predict_proba.return_value = np.array([[0.6, 0.4]])
    
    mock_model.config.id2label = {0: "CLASS_A", 1: "CLASS_B"}
    
    mock_tokenizer = MagicMock()
    
    with patch("transformers.TextClassificationPipeline.__init__", return_value=None):
        pipeline = CalibratedTextClassificationPipeline.__new__(CalibratedTextClassificationPipeline)
        pipeline.model = mock_model
        pipeline.calibrators = [mock_calibrator1, mock_calibrator2]
        
        model_outputs = {"logits": torch.tensor([[1.0, 2.0]])}
        
        # _apply_calibration takes column_stack of calibrator.predict_proba(logits)[:, 1]
        # class 0 prob = 0.8 / (0.8 + 0.4) = 0.8 / 1.2 = 0.666...
        # class 1 prob = 0.4 / (0.8 + 0.4) = 0.4 / 1.2 = 0.333...
        # Wait, the column_stack in the code takes index 1 (positive class) from EACH calibrator.
        # CALIBRATED_PROBS = [cal1.p(logits)[:,1], cal2.p(logits)[:,1]]
        # So class 0 prob = 0.8, class 1 prob = 0.4
        # Then it normalizes:
        # 0.8 / (0.8 + 0.4) = 2/3
        # 0.4 / (0.8 + 0.4) = 1/3
        
        results = pipeline.postprocess(model_outputs, top_k=2)
        
        assert len(results) == 2
        assert results[0]["label"] == "CLASS_A" # 0.666 > 0.333
        assert pytest.approx(results[0]["score"], 0.001) == 0.6666
        assert results[1]["label"] == "CLASS_B"
        assert pytest.approx(results[1]["score"], 0.001) == 0.3333
