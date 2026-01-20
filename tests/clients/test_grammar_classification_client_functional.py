import pytest
import os
from grammared_language.clients.grammar_classification_client import GrammarClassificationClient

# Use a small HF model to minimize download time
TEST_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

RUN_NON_HERMETIC = os.getenv("RUN_NON_HERMETIC", "false").lower() in ("true", "1", "yes")

pytestmark = pytest.mark.skipif(
    not RUN_NON_HERMETIC,
    reason="RUN_NON_HERMETIC=true is not set"
)

class TestGrammarClassificationClientFunctionalHF:
    def test_predict_real_output_single(self):
        client = GrammarClassificationClient(model_id=TEST_MODEL_NAME, backend="hf")
        res = client.predict("This is great!")
        assert isinstance(res, dict)
        assert "label" in res and "score" in res
        assert 0.0 <= res["score"] <= 1.0

    def test_predict_real_output_batch(self):
        client = GrammarClassificationClient(model_id=TEST_MODEL_NAME, backend="hf")
        res = client.predict(["I love this", "I hate this"])
        assert isinstance(res, list)
        assert len(res) == 2
        for r in res:
            assert "label" in r and "score" in r
            assert 0.0 <= r["score"] <= 1.0

# Optional Triton functional tests (requires running Triton server)
try:
    import tritonclient.http as httpclient
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False

@pytest.mark.skipif(
    not TRITON_AVAILABLE or not RUN_NON_HERMETIC,
    reason="Requires tritonclient and RUN_NON_HERMETIC=true (needs running Triton server)"
)
class TestGrammarClassificationClientFunctionalTriton:
    @pytest.fixture(scope="class")
    def triton_ready(self):
        client = httpclient.InferenceServerClient(url="localhost:8000")
        if not client.is_server_live():
            pytest.skip("Triton server is not live")
        return True

    def test_predict_triton_single(self, triton_ready):
        client = GrammarClassificationClient(model_id=TEST_MODEL_NAME, backend="triton", triton_host="localhost", triton_port=8000, triton_model_name="grammared_classifier")
        res = client.predict("This is a good correction.")
        assert isinstance(res, dict)
        assert "label" in res and "score" in res
        assert 0.0 <= res["score"] <= 1.0

    def test_predict_triton_batch(self, triton_ready):
        client = GrammarClassificationClient(model_id=TEST_MODEL_NAME, backend="triton", triton_host="localhost", triton_port=8000, triton_model_name="grammared_classifier")
        res = client.predict(["Good.", "Bad.", "Neutral."])
        assert isinstance(res, list)
        assert len(res) == 3
        for r in res:
            assert "label" in r and "score" in r
            assert 0.0 <= r["score"] <= 1.0
