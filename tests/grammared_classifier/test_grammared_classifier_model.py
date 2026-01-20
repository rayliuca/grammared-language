import pytest
from unittest.mock import MagicMock, patch
from grammared_language.grammared_classifier.grammared_classifier_model import TritonGrammaredClassifierModel
from grammared_language.language_tool.output_models import Match

@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.start_of_replace_token = "<|start_of_replace|>"
    tokenizer.end_of_replace_token = "<|end_of_replace|>"
    tokenizer.sep_token = "<SEP>"
    tokenizer.cls_token = "<CLS>"
    return tokenizer

@pytest.fixture
def mock_pipeline():
    pipeline = MagicMock()
    return pipeline

def test_featurizer_single_match(mock_tokenizer):
    with patch("grammared_language.grammared_classifier.classifier_pipeline.CalibratedTextClassificationPipeline.from_pretrained") as mock_from_pretrained:
        # Prevent actual loading of pretrained model
        mock_from_pretrained.return_value = MagicMock()
        
        model = TritonGrammaredClassifierModel(model_name="dummy_model", tokenizer=mock_tokenizer)
        
        sentence = "I have a apple."
        # Match(offset=9, length=5) for "apple"
        match = Match(offset=9, length=5)
        # Use setattr because Switch allows extra fields but they might not be in __init__
        match.replacement = "an apple"
        
        result = model.featurizer(sentence, [match], 0)
        
        # Expected behavior:
        # result_segments starts with "I have a "
        # original = sentence[9:14] = "apple"
        # replacement = "an apple"
        # correction_segment = "<|start_of_replace|>apple<SEP>an apple<|end_of_replace|>"
        # since i == correction_idx (0 == 0), prepend <CLS>
        # result_segments.append("<CLS><|start_of_replace|>apple<SEP>an apple<|end_of_replace|>")
        # pointer = 9 + 5 = 14
        # trailing text: sentence[14:] = "."
        
        expected = "I have a <CLS><|start_of_replace|>apple<SEP>an apple<|end_of_replace|>."
        assert result == expected

def test_featurizer_multiple_matches(mock_tokenizer):
    with patch("grammared_language.grammared_classifier.classifier_pipeline.CalibratedTextClassificationPipeline.from_pretrained") as mock_from_pretrained:
        mock_from_pretrained.return_value = MagicMock()
        model = TritonGrammaredClassifierModel(model_name="dummy_model", tokenizer=mock_tokenizer)
        
        sentence = "He go to school."
        # Match 1: "go" -> "goes" at offset 3, length 2
        m1 = Match(offset=3, length=2)
        m1.replacement = "goes"
        
        # Match 2: "school" -> "the school" at offset 9, length 6
        m2 = Match(offset=9, length=6)
        m2.replacement = "the school"
        
        # Test selecting first correction (idx 0)
        result0 = model.featurizer(sentence, [m1, m2], 0)
        expected0 = "He <CLS><|start_of_replace|>go<SEP>goes<|end_of_replace|> to <|start_of_replace|>school<SEP>the school<|end_of_replace|>."
        assert result0 == expected0
        
        # Test selecting second correction (idx 1)
        result1 = model.featurizer(sentence, [m1, m2], 1)
        expected1 = "He <|start_of_replace|>go<SEP>goes<|end_of_replace|> to <CLS><|start_of_replace|>school<SEP>the school<|end_of_replace|>."
        assert result1 == expected1

def test_predict_single_sentence(mock_tokenizer, mock_pipeline):
    with patch("grammared_language.grammared_classifier.classifier_pipeline.CalibratedTextClassificationPipeline.from_pretrained", return_value=mock_pipeline):
        model = TritonGrammaredClassifierModel(model_name="dummy_model", tokenizer=mock_tokenizer)
        
        expected_output = [{"label": "label1", "score": 0.9}]
        mock_pipeline.predict.return_value = expected_output
        
        result = model.predict("test sentence")
        
        mock_pipeline.predict.assert_called_once_with("test sentence", top_k=1)
        assert result == expected_output

def test_predict_list_of_sentences(mock_tokenizer, mock_pipeline):
    with patch("grammared_language.grammared_classifier.classifier_pipeline.CalibratedTextClassificationPipeline.from_pretrained", return_value=mock_pipeline):
        model = TritonGrammaredClassifierModel(model_name="dummy_model", tokenizer=mock_tokenizer)
        
        sentences = ["sentence 1", "sentence 2"]
        expected_output = [
            [{"label": "label1", "score": 0.9}],
            [{"label": "label2", "score": 0.8}]
        ]
        mock_pipeline.predict.return_value = expected_output
        
        result = model.predict(sentences)
        
        mock_pipeline.predict.assert_called_once_with(sentences, top_k=1)
        assert result == expected_output

def test_initialization_no_tokenizer():
    with patch("grammared_language.grammared_classifier.classifier_pipeline.CalibratedTextClassificationPipeline.from_pretrained") as mock_pipeline_init, \
         patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer_init:
        
        mock_pipeline_init.return_value = MagicMock()
        mock_tokenizer_init.return_value = MagicMock()
        
        model = TritonGrammaredClassifierModel(model_name="some_model_name")
        
        mock_pipeline_init.assert_called_once_with("some_model_name")
        mock_tokenizer_init.assert_called_once_with("some_model_name")
        assert model.tokenizer == mock_tokenizer_init.return_value
