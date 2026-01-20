from .base_client import BaseClient, LanguageToolRemoteResult
from gector import GECToRTriton, GECToR, predict, load_verb_dict
from transformers import AutoTokenizer
from grammared_language.utils.grammar_correction_extractor import GrammarCorrectionExtractor
# Model initialization
model_id = "gotutiyan/gector-bert-base-cased-5k"
triton_model = GECToRTriton.from_pretrained(model_id, model_name="gector_bert")
tokenizer = AutoTokenizer.from_pretrained(model_id)
encode, decode = load_verb_dict('data/verb-form-vocab.txt')
grammar_correction_extractor = GrammarCorrectionExtractor()

def pred_gector(src: str) -> LanguageToolRemoteResult:
    """
    Perform grammar error correction using GECToR model.
    Args:
        src: Source sentence (string)
    Returns:
        LanguageToolRemoteResult
    """
    corrected = predict(
        triton_model, tokenizer, [src],
        encode, decode,
        keep_confidence=0,
        min_error_prob=0,
        n_iteration=5,
        batch_size=2,
    )
    print(src)
    print(corrected[0])
    matches = grammar_correction_extractor.extract_replacements(src, corrected[0])
    print(matches)
    return LanguageToolRemoteResult(
        language="English",
        languageCode="en-US",
        matches=matches
    )



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
        corrected = self.model.predict([text])[0]
        corrected = predict(
            self.model, self.tokenizer, [text],
            self.encode, self.decode,
            keep_confidence=kwargs.get('keep_confidence', self.pred_config['keep_confidence']),
            min_error_prob=kwargs.get('min_error_prob', self.pred_config['min_error_prob']),
            n_iteration=kwargs.get('n_iteration', self.pred_config['n_iteration']),
            batch_size=kwargs.get('batch_size', self.pred_config['batch_size']),
        )
        return corrected[0]
    
    def _postprocess(self, original, pred, fix_tokenization=True, **kwargs) -> LanguageToolRemoteResult:
        return super()._postprocess(original, pred, fix_tokenization=fix_tokenization, **kwargs)