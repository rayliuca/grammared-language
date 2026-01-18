from ..grammared_classifier.classifier_pipeline import CalibratedTextClassificationPipeline
from ..language_tool.output_models import Match, SuggestedReplacement

class TritonGrammarClassifierModel:
    def __init__(self, model_name: str, tokenizer=None):
        self.pipeline = CalibratedTextClassificationPipeline.from_pretrained(model_name)

        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer = tokenizer

    def featurizer(self, sentence: str, matches: list[Match], correction_idx:int) -> str:
        tokenizer = self.tokenizer
        sentence = sentence
        # Sort corrections in case they're out of order
        # sorted_matches = sorted(matches, key=lambda x: x.offset)
        result_segments = []
        pointer = 0
        for i, m in enumerate(matches):
            # Add unchanged text before correction
            if m.offset > pointer:
                result_segments.append(sentence[pointer:m.offset])

            original = sentence[m.offset:m.offset + m.length]
            replacement = m.replacement
            # Build correction segment
            correction_segment = (
                f"{tokenizer.start_of_replace_token}"
                f"{original}"
                f"{tokenizer.sep_token}"
                f"{replacement}"
                f"{tokenizer.end_of_replace_token}"
            )
            # Prepend cls_token ONLY if this is the selected correction
            if i == correction_idx:
                correction_segment = f"{tokenizer.cls_token}{correction_segment}"
            result_segments.append(correction_segment)
            pointer = m.offset + m.length
        # Add any trailing text after last correction
        if pointer < len(sentence):
            result_segments.append(sentence[pointer:])
        return ''.join(result_segments)

    # def predict(self, sentence, matches: list[Match]) -> list:
    #     features = [self.featurizer(sentence, matches, i) for i in range(len(matches))]
    #     predictions = self.pipeline.predict(features, top_k=1)

    def predict(self, sentence_input: str|list[str]) -> list:
        if isinstance(sentence_input, str):
            sentence_input = [sentence_input]
        return self.pipeline.predict(sentence_input, top_k=1)