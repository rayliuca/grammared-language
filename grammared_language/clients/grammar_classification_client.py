from .base_client import BaseClient
from typing import List, Optional, Union
import json
import numpy as np

try:
    import tritonclient.http as httpclient
    import tritonclient.grpc as grpcclient
    _TRITON_HTTP_AVAILABLE = True
    _TRITON_GRPC_AVAILABLE = True
except Exception:
    try:
        import tritonclient.grpc as grpcclient
        httpclient = None
        _TRITON_HTTP_AVAILABLE = False
        _TRITON_GRPC_AVAILABLE = True
    except Exception:
        httpclient = None
        grpcclient = None
        _TRITON_HTTP_AVAILABLE = False
        _TRITON_GRPC_AVAILABLE = False

_TRITON_AVAILABLE = _TRITON_HTTP_AVAILABLE or _TRITON_GRPC_AVAILABLE

try:
    from transformers import AutoTokenizer
    from grammared_language.grammared_classifier.classifier_pipeline import (
        CalibratedTextClassificationPipeline,
    )
    _TRANSFORMERS_AVAILABLE = True
except Exception:
    AutoTokenizer = None
    CalibratedTextClassificationPipeline = None
    _TRANSFORMERS_AVAILABLE = False

from grammared_language.language_tool.output_models import Match


class GrammarClassificationClient(BaseClient):
    """
    Client for grammar classification.

    Supports two backends:
    - HuggingFace: local `CalibratedTextClassificationPipeline`
    - Triton: remote HTTP inference against a Triton model that returns JSON

    Note: This client overrides `predict` from `BaseClient` to return
    classification outputs (dict or list of dicts) rather than a
    `LanguageToolRemoteResult`.
    """

    def __init__(
        self,
        model_id: str,
        *,
        backend: str = "triton",
        # Triton parameters
        triton_model_name: Optional[str] = None,
        triton_host: str = "localhost",
        triton_port: int = 8001,  # Default to gRPC port
        triton_model_version: Optional[str] = "1",
        triton_protocol: str = "grpc",  # "grpc" or "http"
    ) -> None:
        super().__init__()

        backend = backend.lower()
        if backend not in ("hf", "triton"):
            raise ValueError("backend must be 'hf' or 'triton'")
        self.backend = backend
        self.triton_protocol = triton_protocol.lower()

        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers is required for GrammarClassificationClient. Install with: pip install transformers"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.start_of_replace_token = "<|start_of_replace|>"
        self.tokenizer.end_of_replace_token = "<|end_of_replace|>"

        self.pipeline = None
        self._triton_client = None
        self._triton_model_name = triton_model_name or "grammared_classifier"
        self._triton_model_version = triton_model_version
        self._triton_url = f"{triton_host}:{triton_port}"

        if self.backend == "hf":
            # Initialize tokenizer and calibrated pipeline
            self.pipeline = CalibratedTextClassificationPipeline.from_pretrained(model_id)
        else:
            if not _TRITON_AVAILABLE:
                raise ImportError(
                    "tritonclient is required for Triton backend. Install with: pip install tritonclient[all]"
                )
            
            # Initialize Triton client based on protocol
            if self.triton_protocol == "grpc":
                if not _TRITON_GRPC_AVAILABLE:
                    raise ImportError(
                        "tritonclient.grpc is required for gRPC protocol. Install with: pip install tritonclient[grpc]"
                    )
                self._triton_client = grpcclient.InferenceServerClient(url=self._triton_url)
            else:
                if not _TRITON_HTTP_AVAILABLE:
                    raise ImportError(
                        "tritonclient.http is required for HTTP protocol. Install with: pip install tritonclient[http]"
                    )
                self._triton_client = httpclient.InferenceServerClient(url=self._triton_url)

    def featurizer(self, sentence: str, matches: List[Match], correction_idx: int) -> str:
        """
        Build a single feature string for a given correction index.

        Uses special tokens from the tokenizer:
        - start_of_replace_token, sep_token, end_of_replace_token
        - Prepend cls_token for the selected correction index
        """
        tokenizer = self.tokenizer
        result_segments: List[str] = []
        pointer = 0
        for i, m in enumerate(matches):
            if m.offset > pointer:
                result_segments.append(sentence[pointer:m.offset])

            original = sentence[m.offset : m.offset + m.length]
            replacement = getattr(m, "replacement", None)
            if replacement is None and m.suggestedReplacements and len(m.suggestedReplacements) > 0:
                # Fallback to first suggested replacement text if available
                replacement = m.suggestedReplacements[0].replacement
            if replacement is None:
                replacement = original

            correction_segment = (
                f"{getattr(tokenizer, 'start_of_replace_token', '<|start_of_replace|>')}"
                f"{original}"
                f"{getattr(tokenizer, 'sep_token', '<SEP>')}"
                f"{replacement}"
                f"{getattr(tokenizer, 'end_of_replace_token', '<|end_of_replace|>')}"
            )

            if i == correction_idx:
                correction_segment = f"{getattr(tokenizer, 'cls_token', '<CLS>')}{correction_segment}"

            result_segments.append(correction_segment)
            pointer = m.offset + m.length

        if pointer < len(sentence):
            result_segments.append(sentence[pointer:])
        return "".join(result_segments)

    def predict_matches(self, sentence: str, matches: List[Match]) -> List[dict]:
        """
        Predict classification scores for each correction candidate.

        Returns a list of dicts with keys 'label' and 'score'.
        """
        features = [self.featurizer(sentence, matches, i) for i in range(len(matches))]
        return self.predict(features)  # type: ignore[arg-type]

    def predict(self, texts: Union[str, List[str]]) -> Union[dict, List[dict]]:
        """
        Predict label/score for input text(s).

        - HF backend: uses local calibrated pipeline
        - Triton backend: calls remote Triton HTTP endpoint
        """
        if isinstance(texts, str):
            single = True
            payload = [texts]
        else:
            single = False
            payload = texts

        if self.backend == "hf":
            outputs = self.pipeline.predict(payload, top_k=1)
            return outputs[0] if single else outputs

        # Triton backend
        # Reshape to [batch_size, 1] as expected by the Triton model config
        text_np = np.array(payload, dtype=object).reshape(-1, 1)
        
        # Create inputs based on protocol
        if self.triton_protocol == "grpc":
            inputs = [grpcclient.InferInput("TEXT", list(text_np.shape), "BYTES")]
            inputs[0].set_data_from_numpy(text_np)
            outputs = [grpcclient.InferRequestedOutput("OUTPUT")]
        else:
            inputs = [httpclient.InferInput("TEXT", list(text_np.shape), "BYTES")]
            inputs[0].set_data_from_numpy(text_np)
            outputs = [httpclient.InferRequestedOutput("OUTPUT")]
        
        response = self._triton_client.infer(
            model_name=self._triton_model_name,
            model_version=self._triton_model_version,
            inputs=inputs,
            outputs=outputs,
        )
        out = response.as_numpy("OUTPUT")
        results: List[dict] = []
        for item in out:
            if isinstance(item, bytes):
                s = item.decode("utf-8")
            elif isinstance(item, np.ndarray) and item.dtype.type is np.bytes_:
                s = item.tobytes().decode("utf-8")
            else:
                s = str(item)
            try:
                results.append(json.loads(s))
            except Exception:
                results.append({"label": "unknown", "score": 0.0})

        return results[0] if single else results