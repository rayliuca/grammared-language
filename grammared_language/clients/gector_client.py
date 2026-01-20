from .base_client import BaseClient, LanguageToolRemoteResult
from gector import GECToRTriton, GECToR, predict, load_verb_dict
from transformers import AutoTokenizer
from grammared_language.utils.grammar_correction_extractor import GrammarCorrectionExtractor

class GectorClient(BaseClient):
    def __init__(self, model_id: str, triton_model_name: str=None, verb_dict_path: str='data/verb-form-vocab.txt', **kwargs):
        super().__init__(**kwargs)
        if triton_model_name is None:
            self.model = GECToR.from_pretrained(model_id)
        else:
            self.model = GECToRTriton.from_pretrained(model_id, model_name=triton_model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
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