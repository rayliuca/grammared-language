from pathlib import Path
from urllib.request import urlopen

from .base_client import BaseClient
from grammared_language.language_tool.output_models import LanguageToolRemoteResult
from grammared_language.utils.errant_grammar_correction_extractor import ErrantGrammarCorrectionExtractor

try:
    from gector import GECToR, predict, load_verb_dict
    GECTOR_AVAILABLE = True
except ImportError:
    GECToR = None
    predict = None
    load_verb_dict = None
    GECTOR_AVAILABLE = False

try:
    from gector import GECToRTriton
    GECTOR_TRITON_AVAILABLE = True
except (ImportError, AttributeError):
    GECToRTriton = None
    GECTOR_TRITON_AVAILABLE = False

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    AutoTokenizer = None
    TRANSFORMERS_AVAILABLE = False

OFFICIAL_VOCAB_URL = "https://github.com/grammarly/gector/raw/master/data/verb-form-vocab.txt"
class GectorClient(BaseClient):
    def __init__(
            self,
            pretrained_model_name_or_path: str,
            triton_model_name: str=None,
            triton_host: str='localhost',
            triton_port: int=8001,
            verb_dict_path: str='data/verb-form-vocab.txt',
            auto_download_official_vocab: bool=True,
            **kwargs):
        if "rule_id" not in kwargs:
            kwargs["rule_id"] = triton_model_name or pretrained_model_name_or_path or "Gector"
        super().__init__(**kwargs)
        
        if not GECTOR_AVAILABLE:
            raise ImportError("gector package is required for GectorClient. Install it with: pip install gector")
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers package is required for GectorClient. Install it with: pip install transformers")
        
        if triton_model_name is None:
            self.model = GECToR.from_pretrained(pretrained_model_name_or_path)
        else:
            if GECToRTriton is None:
                raise ImportError("GECToRTriton is not available in your gector installation. Please upgrade gector or use the standard model.")
            self.model = GECToRTriton.from_pretrained(
                pretrained_model_name_or_path, 
                model_name=triton_model_name,
                triton_url=f"{triton_host}:{triton_port}"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        verb_dict_file = Path(verb_dict_path)
        if auto_download_official_vocab and not verb_dict_file.is_file():
            verb_dict_file.parent.mkdir(parents=True, exist_ok=True)
            try:
                with urlopen(OFFICIAL_VOCAB_URL) as response:
                    verb_dict_file.write_bytes(response.read())
            except Exception as exc:
                raise FileNotFoundError(
                    f"Failed to download verb vocab from {OFFICIAL_VOCAB_URL} to {verb_dict_file}"
                ) from exc

        self.encode, self.decode = load_verb_dict(verb_dict_path)

        self.pred_config = {
            'keep_confidence': kwargs.get('keep_confidence', 0),
            'min_error_prob': kwargs.get('min_error_prob', 0),
            'n_iteration': kwargs.get('n_iteration', 5),
            'batch_size': kwargs.get('batch_size', 2),
        }

    def _predict(self, text: str|list[str], **kwargs) -> str|list[str]:
        if isinstance(text, list):
            corrected = predict(
                self.model, self.tokenizer, text,
                self.encode, self.decode,
                keep_confidence=kwargs.get('keep_confidence', self.pred_config['keep_confidence']),
                min_error_prob=kwargs.get('min_error_prob', self.pred_config['min_error_prob']),
                n_iteration=kwargs.get('n_iteration', self.pred_config['n_iteration']),
                batch_size=kwargs.get('batch_size', self.pred_config['batch_size']),
            )
            return corrected
        
        corrected = predict(
            self.model, self.tokenizer, [text],
            self.encode, self.decode,
            keep_confidence=kwargs.get('keep_confidence', self.pred_config['keep_confidence']),
            min_error_prob=kwargs.get('min_error_prob', self.pred_config['min_error_prob']),
            n_iteration=kwargs.get('n_iteration', self.pred_config['n_iteration']),
            batch_size=kwargs.get('batch_size', self.pred_config['batch_size']),
        )
        return corrected[0]