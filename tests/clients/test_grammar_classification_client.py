import pytest
from unittest.mock import MagicMock, patch
from grammared_language.clients.grammar_classification_client import GrammarClassificationClient
from grammared_language.language_tool.output_models import Match


class TestGrammarClassificationClientUnit:
    def test_featurizer_single_match(self):
        mock_tokenizer = MagicMock()
        mock_tokenizer.start_of_replace_token = "<|start_of_replace|>"
        mock_tokenizer.end_of_replace_token = "<|end_of_replace|>"
        mock_tokenizer.sep_token = "<SEP>"
        mock_tokenizer.cls_token = "<CLS>"

        with patch("grammared_language.clients.grammar_classification_client.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
             patch("grammared_language.clients.grammar_classification_client.CalibratedTextClassificationPipeline.from_pretrained", return_value=MagicMock()):
            client = GrammarClassificationClient(model_id="dummy_model", backend="hf")

        sentence = "I have a apple."
        match = Match(offset=9, length=5)
        setattr(match, "replacement", "an apple")

        result = client.featurizer(sentence, [match], 0)
        expected = "I have a <CLS><|start_of_replace|>apple<SEP>an apple<|end_of_replace|>."
        assert result == expected

    def test_featurizer_multiple_matches(self):
        mock_tokenizer = MagicMock()
        mock_tokenizer.start_of_replace_token = "<|start_of_replace|>"
        mock_tokenizer.end_of_replace_token = "<|end_of_replace|>"
        mock_tokenizer.sep_token = "<SEP>"
        mock_tokenizer.cls_token = "<CLS>"

        with patch("grammared_language.clients.grammar_classification_client.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
             patch("grammared_language.clients.grammar_classification_client.CalibratedTextClassificationPipeline.from_pretrained", return_value=MagicMock()):
            client = GrammarClassificationClient(model_id="dummy_model", backend="hf")

        sentence = "He go to school."
        m1 = Match(offset=3, length=2)
        setattr(m1, "replacement", "goes")
        m2 = Match(offset=9, length=6)
        setattr(m2, "replacement", "the school")

        result0 = client.featurizer(sentence, [m1, m2], 0)
        expected0 = "He <CLS><|start_of_replace|>go<SEP>goes<|end_of_replace|> to <|start_of_replace|>school<SEP>the school<|end_of_replace|>."
        assert result0 == expected0

        result1 = client.featurizer(sentence, [m1, m2], 1)
        expected1 = "He <|start_of_replace|>go<SEP>goes<|end_of_replace|> to <CLS><|start_of_replace|>school<SEP>the school<|end_of_replace|>."
        assert result1 == expected1

    def test_predict_hf_single_and_batch(self):
        mock_tokenizer = MagicMock()
        mock_pipeline = MagicMock()

        with patch("grammared_language.clients.grammar_classification_client.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
             patch("grammared_language.clients.grammar_classification_client.CalibratedTextClassificationPipeline.from_pretrained", return_value=mock_pipeline):
            client = GrammarClassificationClient(model_id="dummy_model", backend="hf")

        # Single text
        expected_single = [{"label": "POS", "score": 0.9}]
        mock_pipeline.predict.return_value = expected_single
        out_single = client.predict("hello")
        mock_pipeline.predict.assert_called_with(["hello"], top_k=1)
        assert out_single == expected_single[0]

        # Batch texts
        expected_batch = [
            [{"label": "POS", "score": 0.9}],
            [{"label": "NEG", "score": 0.8}],
        ]
        mock_pipeline.predict.return_value = expected_batch
        out_batch = client.predict(["hello", "world"])
        mock_pipeline.predict.assert_called_with(["hello", "world"], top_k=1)
        assert out_batch == expected_batch

    def test_predict_matches_uses_featurizer_and_predict(self):
        mock_tokenizer = MagicMock()
        mock_pipeline = MagicMock()

        with patch("grammared_language.clients.grammar_classification_client.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
             patch("grammared_language.clients.grammar_classification_client.CalibratedTextClassificationPipeline.from_pretrained", return_value=mock_pipeline):
            client = GrammarClassificationClient(model_id="dummy_model", backend="hf")

        sentence = "Text"
        m1 = Match(offset=0, length=0)
        setattr(m1, "replacement", "A")
        m2 = Match(offset=0, length=0)
        setattr(m2, "replacement", "B")

        # Stub predict to return list of dicts matching features length
        # HF pipeline returns list of dicts (one per input)
        mock_pipeline.predict.return_value = [
            {"label": "L1", "score": 0.7},
            {"label": "L2", "score": 0.3},
        ]
        out = client.predict_matches(sentence, [m1, m2])
        assert isinstance(out, list)
        assert len(out) == 2
        assert {"label", "score"}.issubset(out[0].keys())

    def test_predict_triton_parses_json_bytes(self, monkeypatch):
        # Build lightweight grpcclient stub
        class StubInferInput:
            def __init__(self, name, shape, dtype):
                self.name = name
                self.shape = shape
                self.dtype = dtype
            def set_data_from_numpy(self, arr):
                self.arr = arr
        class StubInferRequestedOutput:
            def __init__(self, name):
                self.name = name
        class StubResponse:
            def __init__(self, objs):
                self._objs = objs
            def as_numpy(self, name):
                return self._objs
        class StubClient:
            def __init__(self, url):
                self.url = url
            def infer(self, **kwargs):
                # Simulate Triton returning JSON bytes per input
                return StubResponse(__import__("numpy").array([
                    b'{"label": "none", "score": 0.55}',
                    b'{"label": "none", "score": 0.45}',
                ], dtype=object))
        stub_grpcclient = MagicMock()
        stub_grpcclient.InferInput = StubInferInput
        stub_grpcclient.InferRequestedOutput = StubInferRequestedOutput
        stub_grpcclient.InferenceServerClient = StubClient

        monkeypatch.setattr(
            "grammared_language.clients.grammar_classification_client.grpcclient",
            stub_grpcclient,
        )

        # Patch transformers to avoid real load
        mock_tokenizer = MagicMock()
        with patch("grammared_language.clients.grammar_classification_client.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):
            client = GrammarClassificationClient(model_id="dummy_model", backend="triton", triton_host="localhost", triton_port=8001)

        out = client.predict(["a", "b"])
        assert isinstance(out, list)
        assert len(out) == 2
        assert out[0]["label"] == "none"
        assert 0.0 <= out[0]["score"] <= 1.0
