from __future__ import annotations
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field


Difficulty = Literal["easy", "medium", "tricky", "higher"]


class GenerationConfig(BaseModel):
    prompt_template: str
    difficulty: Difficulty = "medium"
    include_units: bool = False
    include_subunits: bool = False
    picked_topic: Optional[str] = None


class ValidationCriteria(BaseModel):
    # Simple starter criteria; extend later
    must_cover_units: Optional[List[str]] = None
    language: Literal["sr", "en"] = "sr"
    max_length_tokens: Optional[int] = None


class Sample(BaseModel):
    id: str
    curriculum_paths: List[str] = Field(default_factory=list)
    # e.g. pointers to files under data/


class GenerationInput(BaseModel):
    sample: Sample
    config: GenerationConfig


class GenerationOutput(BaseModel):
    sample_id: str
    prompt: str
    completion: str
    meta: Dict[str, Any] = Field(default_factory=dict)


class ValidationInput(BaseModel):
    generation: GenerationOutput
    criteria: ValidationCriteria


class ValidationOutput(BaseModel):
    sample_id: str
    score: float
    verdict: Literal["pass", "fail"]
    feedback: str
    meta: Dict[str, Any] = Field(default_factory=dict)
