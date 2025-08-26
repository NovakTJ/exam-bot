from __future__ import annotations
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field


"""Core data models used across the exam generation/validation pipeline."""

Difficulty = Literal["easy", "medium", "tricky", "higher"]


class GenerationConfig(BaseModel):
    """Tuning knobs for generation phase.

    Attributes:
        prompt_template: Name or type of the prompt strategy (e.g., "one-shot").
        difficulty: Overall difficulty level.
        include_units: Whether to include unit snippets in prompt.
        include_subunits: Whether to include subunit hints.
        picked_topic: Optional specific topic focus.
    """
    prompt_template: str
    difficulty: Difficulty = "medium"
    include_units: bool = False
    include_subunits: bool = False
    picked_topic: Optional[str] = None


class ValidationCriteria(BaseModel):
    """High-level validation constraints.

    Attributes:
        must_cover_units: List of unit names/ids expected to be covered.
        language: Expected language code ("sr" or "en").
        max_length_tokens: Soft cap on response length.
    """
    must_cover_units: Optional[List[str]] = None
    language: Literal["sr", "en"] = "sr"
    max_length_tokens: Optional[int] = None


class Sample(BaseModel):
    """A single test case binding an id to curriculum file paths.

    Example:
        >>> Sample(id="s001", curriculum_paths=["data/page_0001_extracted_text.txt"])  # doctest: +ELLIPSIS
        Sample(id='s001', curriculum_paths=['data/page_0001_extracted_text.txt'])
    """
    id: str
    curriculum_paths: List[str] = Field(default_factory=list)
    # e.g. pointers to files under data/


class GenerationInput(BaseModel):
    """Input bundle for generation, pairing a Sample with a config."""
    sample: Sample
    config: GenerationConfig


class GenerationOutput(BaseModel):
    """LLM output of generation, plus the rendered prompt for traceability."""
    sample_id: str
    prompt: str
    completion: str
    meta: Dict[str, Any] = Field(default_factory=dict)


class ValidationInput(BaseModel):
    """Input bundle for validation, pairing generated output and criteria."""
    generation: GenerationOutput
    criteria: ValidationCriteria


class ValidationOutput(BaseModel):
    """Result of validator LLM.

    Attributes:
        sample_id: Which sample this evaluation refers to.
        score: Value in [0,1].
        verdict: "pass" or "fail".
        feedback: Short rationale.
        meta: Optional extra metadata.
    """
    sample_id: str
    score: float
    verdict: Literal["pass", "fail"]
    feedback: str
    meta: Dict[str, Any] = Field(default_factory=dict)
