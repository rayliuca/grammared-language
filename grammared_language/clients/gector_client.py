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

class GectorClient(BaseClient):
    def __init__(self, pretrained_model_name_or_path: str, triton_model_name: str=None, verb_dict_path: str='data/verb-form-vocab.txt', **kwargs):
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
            self.model = GECToRTriton.from_pretrained(pretrained_model_name_or_path, model_name=triton_model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.encode, self.decode = load_verb_dict(verb_dict_path)

        self.pred_config = {
            'keep_confidence': kwargs.get('keep_confidence', 0),
            'min_error_prob': kwargs.get('min_error_prob', 0),
            'n_iteration': kwargs.get('n_iteration', 5),
            'batch_size': kwargs.get('batch_size', 2),
        }

    def _predict(self, text: str, **kwargs) -> str:
        corrected = predict(
            self.model, self.tokenizer, [text],
            self.encode, self.decode,
            keep_confidence=kwargs.get('keep_confidence', self.pred_config['keep_confidence']),
            min_error_prob=kwargs.get('min_error_prob', self.pred_config['min_error_prob']),
            n_iteration=kwargs.get('n_iteration', self.pred_config['n_iteration']),
            batch_size=kwargs.get('batch_size', self.pred_config['batch_size']),
        )
        return corrected[0]