from typing import Optional, List, Any
from pydantic import BaseModel, Field, ConfigDict

class Software(BaseModel):
    name: Optional[str] = None
    version: Optional[str] = None
    
    model_config = ConfigDict(extra="allow")


class IgnoreRange(BaseModel):
    offset: int = Field(ge=0)
    length: int = Field(ge=0)


class Replacement(BaseModel):
    value: str


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
    id: str
    description: str
    subId: Optional[str] = None
    issueType: Optional[str] = None
    urls: Optional[List[RuleUrl]] = None
    category: Optional[RuleCategory] = None
    tags: Optional[List[RuleTag]] = None
    
    model_config = ConfigDict(extra="allow")


class Match(BaseModel):
    message: str
    context: Context
    offset: int = Field(ge=0)
    length: int = Field(ge=0)
    rule: Rule
    shortMessage: Optional[str] = None
    replacements: Optional[List[Replacement]] = None
    
    model_config = ConfigDict(extra="allow")


class LanguageToolRemoteResult(BaseModel):
    language: str
    languageCode: str
    matches: List[Match]
    languageDetectedCode: Optional[str] = None
    languageDetectedName: Optional[str] = None
    software: Optional[Software] = None
    ignoreRanges: Optional[List[IgnoreRange]] = None
    
    model_config = ConfigDict(extra="allow")