from typing import Optional, List
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum

class SuggestionType(int, Enum):
    Default = 0
    Translation = 1
    Curated = 2

class MatchType(int, Enum):
    UnknownWord = 0
    Hint = 1
    Other = 2

class Tag(int, Enum):
    picky = 0
    academic = 1
    clarity = 2
    professional = 3
    creative = 4
    customer = 5
    jobapp = 6
    objective = 7
    elegant = 8

class Software(BaseModel):
    name: Optional[str] = None
    version: Optional[str] = None
    
    model_config = ConfigDict(extra="allow", from_attributes=True)


class IgnoreRange(BaseModel):
    offset: int = Field(ge=0)
    length: int = Field(ge=0)


class SuggestedReplacement(BaseModel):
    replacement: str = ""
    description: Optional[str] = None
    suffix: Optional[str] = None  # Value shown in the UI after the replacement (but not part of it).
    confidence: Optional[float] = None  # from 0 (lowest) to 1 (highest)
    type: SuggestionType = SuggestionType.Default
    model_config = ConfigDict(from_attributes=True)


class Context(BaseModel):
    text: str
    offset: int = Field(ge=0)
    length: int = Field(ge=0)


class RuleCategory(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None


class RuleTag(BaseModel):
    name: str
    value: str


class RuleUrl(BaseModel):
    value: str = Field(pattern=r"^https?://")  # URI format


class Rule(BaseModel):
    id: Optional[str] = None
    description: Optional[str] = None
    subId: Optional[str] = None
    issueType: Optional[str] = None
    urls: Optional[List[RuleUrl]] = None
    category: Optional[RuleCategory] = None
    tags: Optional[List[Tag]] = None
    sourceFile: Optional[str] = None
    tempOff: Optional[bool] = None
    isPremium: Optional[bool] = None

    model_config = ConfigDict(extra="allow", from_attributes=True)


class Match(BaseModel):
    offset: int = Field(ge=0)
    length: int = Field(ge=0)
    id: Optional[str] = None
    sub_id: Optional[str] = None
    suggestions: Optional[List[str]] = None
    ruleDescription: Optional[str] = None
    matchDescription: Optional[str] = None
    matchShortDescription: Optional[str] = None
    url: Optional[str] = None
    suggestedReplacements: Optional[List[SuggestedReplacement]] = None
    autoCorrect: Optional[bool] = None
    type: Optional[MatchType] = None
    contextForSureMatch: Optional[int] = None
    rule: Optional[Rule] = None
    # For backward compatibility with previous fields
    message: Optional[str] = None
    shortMessage: Optional[str] = None
    suggested_replacements: Optional[List[SuggestedReplacement]] = None
    
    model_config = ConfigDict(extra="allow", from_attributes=True)


class LanguageToolRemoteResult(BaseModel):
    language: str
    languageCode: str
    matches: List[Match]
    languageDetectedCode: Optional[str] = None
    languageDetectedName: Optional[str] = None
    software: Optional[Software] = None
    ignoreRanges: Optional[List[IgnoreRange]] = None
    
    model_config = ConfigDict(extra="allow", from_attributes=True)