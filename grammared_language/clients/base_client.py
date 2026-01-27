from grammared_language.language_tool.output_models import LanguageToolRemoteResult
from grammared_language.utils.errant_grammar_correction_extractor import ErrantGrammarCorrectionExtractor

class BaseClient:
    def __init__(self, *args, **kwargs):
        self.rule_id = kwargs.get("rule_id", "GrammaredLanguage")
        print(f"Initialized BaseClient with rule_id: {self.rule_id}")
        self.correction_extractor = ErrantGrammarCorrectionExtractor(rule_id=self.rule_id)
    
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

    def predict(self, text: str|list[str]) -> LanguageToolRemoteResult|list[LanguageToolRemoteResult]:
        single = True
        if isinstance(text, list):
            single = False

        if single:
            _text = self._preprocess(text)
        else:
            _text = [self._preprocess(t) for t in text]
        corrected_text: str|list[str] = self._predict(_text)

        output = None
        if single:
            pred: LanguageToolRemoteResult = self._pred_postprocess(text, corrected_text)
            output = self._output_postprocess(text, pred)
        else:
            pred = [
                self._pred_postprocess(orig, corr)
                for orig, corr in zip(text, corrected_text)
            ]
            output = [
                self._output_postprocess(orig, p)
                for orig, p in zip(text, pred)
            ]
        return output

    def __call__(self, text: str) -> LanguageToolRemoteResult:
        return self.predict(text)