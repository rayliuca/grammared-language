from grammared_language.language_tool.output_models import LanguageToolRemoteResult
from grammared_language.utils.errant_grammar_correction_extractor import ErrantGrammarCorrectionExtractor

class BaseClient:
    def __init__(self, *args, **kwargs):
        self.correction_extractor = ErrantGrammarCorrectionExtractor()
    
    def _preprocess(self, text: str) -> str:
        return text
    
    def _predict(self, text: str) -> str:
        raise NotImplementedError("Subclasses should implement this method.")
    
    def _pred_postprocess(self, original: str, pred: str, **kwargs) -> LanguageToolRemoteResult:
        matches = self.correction_extractor.extract_replacements(
            original=original, corrected=pred, fix_tokenization=kwargs.get('fix_tokenization', True)
        )
        return LanguageToolRemoteResult(
            language="English",
            languageCode="en-US",
            matches=matches
        )
    
    def _output_postprocess(self, original: str, pred: LanguageToolRemoteResult, **kwargs) -> LanguageToolRemoteResult:
        return pred

    def predict(self, text: str) -> LanguageToolRemoteResult:
        _text = self._preprocess(text)
        corrected_text: str = self._predict(_text)
        pred: LanguageToolRemoteResult = self._pred_postprocess(text, corrected_text)
        return self._output_postprocess(text, pred)

    def __call__(self, text: str) -> LanguageToolRemoteResult:
        return self.predict(text)