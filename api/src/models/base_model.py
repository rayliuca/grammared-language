from ..util import LanguageToolRemoteResult, GrammarCorrectionExtractor

class BaseModel:
    def __init__(self, *args, **kwargs):
        self.correction_extractor = GrammarCorrectionExtractor()
    
    def _preprocess(self, text: str) -> str:
        return text
    
    def _predict(self, text: str) -> str:
        raise NotImplementedError("Subclasses should implement this method.")
    
    def _pred_postprocess(self, original: str, pred: str, **kwargs) -> LanguageToolRemoteResult:
        return self.correction_extractor.extract_replacements(
            original=original, corrected=pred, fix_tokenization=kwargs.get('fix_tokenization', True)
        )
    
    def _output_postprocess(self, original: str, pred: LanguageToolRemoteResult, **kwargs) -> LanguageToolRemoteResult:
        return pred

    def predict(self, text: str) -> LanguageToolRemoteResult:
        text = self._preprocess(text)
        corrected_text: str = self._predict(text)
        pred: LanguageToolRemoteResult = self._pred_postprocess(text, corrected_text)
        return self._output_postprocess(text, pred)
    