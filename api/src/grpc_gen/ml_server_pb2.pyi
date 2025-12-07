from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class AnalyzeRequest(_message.Message):
    __slots__ = ("text", "options")
    TEXT_FIELD_NUMBER: _ClassVar[int]
    OPTIONS_FIELD_NUMBER: _ClassVar[int]
    text: str
    options: ProcessingOptions
    def __init__(self, text: _Optional[str] = ..., options: _Optional[_Union[ProcessingOptions, _Mapping]] = ...) -> None: ...

class ProcessingOptions(_message.Message):
    __slots__ = ("language", "tempOff", "level", "premium", "enabledOnly", "enabledRules", "disabledRules")
    class Level(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        picky: _ClassVar[ProcessingOptions.Level]
        academic: _ClassVar[ProcessingOptions.Level]
        clarity: _ClassVar[ProcessingOptions.Level]
        professional: _ClassVar[ProcessingOptions.Level]
        creative: _ClassVar[ProcessingOptions.Level]
        customer: _ClassVar[ProcessingOptions.Level]
        jobapp: _ClassVar[ProcessingOptions.Level]
        objective: _ClassVar[ProcessingOptions.Level]
        elegant: _ClassVar[ProcessingOptions.Level]
        defaultLevel: _ClassVar[ProcessingOptions.Level]
    picky: ProcessingOptions.Level
    academic: ProcessingOptions.Level
    clarity: ProcessingOptions.Level
    professional: ProcessingOptions.Level
    creative: ProcessingOptions.Level
    customer: ProcessingOptions.Level
    jobapp: ProcessingOptions.Level
    objective: ProcessingOptions.Level
    elegant: ProcessingOptions.Level
    defaultLevel: ProcessingOptions.Level
    LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    TEMPOFF_FIELD_NUMBER: _ClassVar[int]
    LEVEL_FIELD_NUMBER: _ClassVar[int]
    PREMIUM_FIELD_NUMBER: _ClassVar[int]
    ENABLEDONLY_FIELD_NUMBER: _ClassVar[int]
    ENABLEDRULES_FIELD_NUMBER: _ClassVar[int]
    DISABLEDRULES_FIELD_NUMBER: _ClassVar[int]
    language: str
    tempOff: bool
    level: ProcessingOptions.Level
    premium: bool
    enabledOnly: bool
    enabledRules: _containers.RepeatedScalarFieldContainer[str]
    disabledRules: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, language: _Optional[str] = ..., tempOff: bool = ..., level: _Optional[_Union[ProcessingOptions.Level, str]] = ..., premium: bool = ..., enabledOnly: bool = ..., enabledRules: _Optional[_Iterable[str]] = ..., disabledRules: _Optional[_Iterable[str]] = ...) -> None: ...

class AnalyzeResponse(_message.Message):
    __slots__ = ("sentences",)
    SENTENCES_FIELD_NUMBER: _ClassVar[int]
    sentences: _containers.RepeatedCompositeFieldContainer[AnalyzedSentence]
    def __init__(self, sentences: _Optional[_Iterable[_Union[AnalyzedSentence, _Mapping]]] = ...) -> None: ...

class ProcessRequest(_message.Message):
    __slots__ = ("sentences", "options")
    SENTENCES_FIELD_NUMBER: _ClassVar[int]
    OPTIONS_FIELD_NUMBER: _ClassVar[int]
    sentences: _containers.RepeatedCompositeFieldContainer[AnalyzedSentence]
    options: ProcessingOptions
    def __init__(self, sentences: _Optional[_Iterable[_Union[AnalyzedSentence, _Mapping]]] = ..., options: _Optional[_Union[ProcessingOptions, _Mapping]] = ...) -> None: ...

class ProcessResponse(_message.Message):
    __slots__ = ("rawMatches", "matches")
    RAWMATCHES_FIELD_NUMBER: _ClassVar[int]
    MATCHES_FIELD_NUMBER: _ClassVar[int]
    rawMatches: _containers.RepeatedCompositeFieldContainer[Match]
    matches: _containers.RepeatedCompositeFieldContainer[Match]
    def __init__(self, rawMatches: _Optional[_Iterable[_Union[Match, _Mapping]]] = ..., matches: _Optional[_Iterable[_Union[Match, _Mapping]]] = ...) -> None: ...

class AnalyzedMatchRequest(_message.Message):
    __slots__ = ("sentences", "inputLogging", "textSessionID")
    SENTENCES_FIELD_NUMBER: _ClassVar[int]
    INPUTLOGGING_FIELD_NUMBER: _ClassVar[int]
    TEXTSESSIONID_FIELD_NUMBER: _ClassVar[int]
    sentences: _containers.RepeatedCompositeFieldContainer[AnalyzedSentence]
    inputLogging: bool
    textSessionID: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, sentences: _Optional[_Iterable[_Union[AnalyzedSentence, _Mapping]]] = ..., inputLogging: bool = ..., textSessionID: _Optional[_Iterable[int]] = ...) -> None: ...

class AnalyzedSentence(_message.Message):
    __slots__ = ("text", "tokens")
    TEXT_FIELD_NUMBER: _ClassVar[int]
    TOKENS_FIELD_NUMBER: _ClassVar[int]
    text: str
    tokens: _containers.RepeatedCompositeFieldContainer[AnalyzedTokenReadings]
    def __init__(self, text: _Optional[str] = ..., tokens: _Optional[_Iterable[_Union[AnalyzedTokenReadings, _Mapping]]] = ...) -> None: ...

class AnalyzedTokenReadings(_message.Message):
    __slots__ = ("readings", "chunkTags", "startPos")
    READINGS_FIELD_NUMBER: _ClassVar[int]
    CHUNKTAGS_FIELD_NUMBER: _ClassVar[int]
    STARTPOS_FIELD_NUMBER: _ClassVar[int]
    readings: _containers.RepeatedCompositeFieldContainer[AnalyzedToken]
    chunkTags: _containers.RepeatedScalarFieldContainer[str]
    startPos: int
    def __init__(self, readings: _Optional[_Iterable[_Union[AnalyzedToken, _Mapping]]] = ..., chunkTags: _Optional[_Iterable[str]] = ..., startPos: _Optional[int] = ...) -> None: ...

class AnalyzedToken(_message.Message):
    __slots__ = ("token", "posTag", "lemma")
    TOKEN_FIELD_NUMBER: _ClassVar[int]
    POSTAG_FIELD_NUMBER: _ClassVar[int]
    LEMMA_FIELD_NUMBER: _ClassVar[int]
    token: str
    posTag: str
    lemma: str
    def __init__(self, token: _Optional[str] = ..., posTag: _Optional[str] = ..., lemma: _Optional[str] = ...) -> None: ...

class PostProcessingRequest(_message.Message):
    __slots__ = ("sentences", "matches", "inputLogging", "textSessionID")
    SENTENCES_FIELD_NUMBER: _ClassVar[int]
    MATCHES_FIELD_NUMBER: _ClassVar[int]
    INPUTLOGGING_FIELD_NUMBER: _ClassVar[int]
    TEXTSESSIONID_FIELD_NUMBER: _ClassVar[int]
    sentences: _containers.RepeatedScalarFieldContainer[str]
    matches: _containers.RepeatedCompositeFieldContainer[MatchList]
    inputLogging: bool
    textSessionID: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, sentences: _Optional[_Iterable[str]] = ..., matches: _Optional[_Iterable[_Union[MatchList, _Mapping]]] = ..., inputLogging: bool = ..., textSessionID: _Optional[_Iterable[int]] = ...) -> None: ...

class MatchRequest(_message.Message):
    __slots__ = ("sentences", "inputLogging", "textSessionID")
    SENTENCES_FIELD_NUMBER: _ClassVar[int]
    INPUTLOGGING_FIELD_NUMBER: _ClassVar[int]
    TEXTSESSIONID_FIELD_NUMBER: _ClassVar[int]
    sentences: _containers.RepeatedScalarFieldContainer[str]
    inputLogging: bool
    textSessionID: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, sentences: _Optional[_Iterable[str]] = ..., inputLogging: bool = ..., textSessionID: _Optional[_Iterable[int]] = ...) -> None: ...

class MatchResponse(_message.Message):
    __slots__ = ("sentenceMatches",)
    SENTENCEMATCHES_FIELD_NUMBER: _ClassVar[int]
    sentenceMatches: _containers.RepeatedCompositeFieldContainer[MatchList]
    def __init__(self, sentenceMatches: _Optional[_Iterable[_Union[MatchList, _Mapping]]] = ...) -> None: ...

class MatchList(_message.Message):
    __slots__ = ("matches",)
    MATCHES_FIELD_NUMBER: _ClassVar[int]
    matches: _containers.RepeatedCompositeFieldContainer[Match]
    def __init__(self, matches: _Optional[_Iterable[_Union[Match, _Mapping]]] = ...) -> None: ...

class Match(_message.Message):
    __slots__ = ("offset", "length", "id", "sub_id", "suggestions", "ruleDescription", "matchDescription", "matchShortDescription", "url", "suggestedReplacements", "autoCorrect", "type", "contextForSureMatch", "rule")
    class MatchType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        UnknownWord: _ClassVar[Match.MatchType]
        Hint: _ClassVar[Match.MatchType]
        Other: _ClassVar[Match.MatchType]
    UnknownWord: Match.MatchType
    Hint: Match.MatchType
    Other: Match.MatchType
    OFFSET_FIELD_NUMBER: _ClassVar[int]
    LENGTH_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    SUB_ID_FIELD_NUMBER: _ClassVar[int]
    SUGGESTIONS_FIELD_NUMBER: _ClassVar[int]
    RULEDESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    MATCHDESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    MATCHSHORTDESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    URL_FIELD_NUMBER: _ClassVar[int]
    SUGGESTEDREPLACEMENTS_FIELD_NUMBER: _ClassVar[int]
    AUTOCORRECT_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    CONTEXTFORSUREMATCH_FIELD_NUMBER: _ClassVar[int]
    RULE_FIELD_NUMBER: _ClassVar[int]
    offset: int
    length: int
    id: str
    sub_id: str
    suggestions: _containers.RepeatedScalarFieldContainer[str]
    ruleDescription: str
    matchDescription: str
    matchShortDescription: str
    url: str
    suggestedReplacements: _containers.RepeatedCompositeFieldContainer[SuggestedReplacement]
    autoCorrect: bool
    type: Match.MatchType
    contextForSureMatch: int
    rule: Rule
    def __init__(self, offset: _Optional[int] = ..., length: _Optional[int] = ..., id: _Optional[str] = ..., sub_id: _Optional[str] = ..., suggestions: _Optional[_Iterable[str]] = ..., ruleDescription: _Optional[str] = ..., matchDescription: _Optional[str] = ..., matchShortDescription: _Optional[str] = ..., url: _Optional[str] = ..., suggestedReplacements: _Optional[_Iterable[_Union[SuggestedReplacement, _Mapping]]] = ..., autoCorrect: bool = ..., type: _Optional[_Union[Match.MatchType, str]] = ..., contextForSureMatch: _Optional[int] = ..., rule: _Optional[_Union[Rule, _Mapping]] = ...) -> None: ...

class Rule(_message.Message):
    __slots__ = ("sourceFile", "issueType", "tempOff", "category", "isPremium", "tags")
    class Tag(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        picky: _ClassVar[Rule.Tag]
        academic: _ClassVar[Rule.Tag]
        clarity: _ClassVar[Rule.Tag]
        professional: _ClassVar[Rule.Tag]
        creative: _ClassVar[Rule.Tag]
        customer: _ClassVar[Rule.Tag]
        jobapp: _ClassVar[Rule.Tag]
        objective: _ClassVar[Rule.Tag]
        elegant: _ClassVar[Rule.Tag]
    picky: Rule.Tag
    academic: Rule.Tag
    clarity: Rule.Tag
    professional: Rule.Tag
    creative: Rule.Tag
    customer: Rule.Tag
    jobapp: Rule.Tag
    objective: Rule.Tag
    elegant: Rule.Tag
    SOURCEFILE_FIELD_NUMBER: _ClassVar[int]
    ISSUETYPE_FIELD_NUMBER: _ClassVar[int]
    TEMPOFF_FIELD_NUMBER: _ClassVar[int]
    CATEGORY_FIELD_NUMBER: _ClassVar[int]
    ISPREMIUM_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    sourceFile: str
    issueType: str
    tempOff: bool
    category: RuleCategory
    isPremium: bool
    tags: _containers.RepeatedScalarFieldContainer[Rule.Tag]
    def __init__(self, sourceFile: _Optional[str] = ..., issueType: _Optional[str] = ..., tempOff: bool = ..., category: _Optional[_Union[RuleCategory, _Mapping]] = ..., isPremium: bool = ..., tags: _Optional[_Iterable[_Union[Rule.Tag, str]]] = ...) -> None: ...

class RuleCategory(_message.Message):
    __slots__ = ("id", "name")
    ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    id: str
    name: str
    def __init__(self, id: _Optional[str] = ..., name: _Optional[str] = ...) -> None: ...

class SuggestedReplacement(_message.Message):
    __slots__ = ("replacement", "description", "suffix", "confidence", "type")
    class SuggestionType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        Default: _ClassVar[SuggestedReplacement.SuggestionType]
        Translation: _ClassVar[SuggestedReplacement.SuggestionType]
        Curated: _ClassVar[SuggestedReplacement.SuggestionType]
    Default: SuggestedReplacement.SuggestionType
    Translation: SuggestedReplacement.SuggestionType
    Curated: SuggestedReplacement.SuggestionType
    REPLACEMENT_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    SUFFIX_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    replacement: str
    description: str
    suffix: str
    confidence: float
    type: SuggestedReplacement.SuggestionType
    def __init__(self, replacement: _Optional[str] = ..., description: _Optional[str] = ..., suffix: _Optional[str] = ..., confidence: _Optional[float] = ..., type: _Optional[_Union[SuggestedReplacement.SuggestionType, str]] = ...) -> None: ...
