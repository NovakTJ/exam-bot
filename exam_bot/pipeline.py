from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any
from pydantic import BaseModel

from .types import (
    GenerationInput,
    GenerationOutput,
    ValidationInput,
    ValidationOutput,
)
from .anthropic_utils import (
    create_messages_payload,
    write_batch_items,
    run_batch,
    poll_batch,
    download_batch_results,
    parse_ndjson,
    extract_text_from_response,
    call_messages,
    upload_file,
)


class Prompts(BaseModel):
    """Convenience holder for system prompts if needed in the future."""
    system_generate: str
    system_validate: str


def render_generation_prompt(gi: GenerationInput, curriculum_text: str, base_prompt: str | None = None) -> str:
    """Render the user content for generation.

    Args:
        gi: Generation input with sample and config.
        curriculum_text: Concatenated snippet text from curriculum files.
        base_prompt: Optional preamble to prepend.

    Returns:
        String to be used as user content in messages.create.
    """
    parts: List[str] = []
    if base_prompt:
        parts.append(base_prompt.strip())
    else:
        parts.append("You are an esteemed OOP1 professor. Generate an exam in Serbian.")
    parts.append(f"Difficulty: {gi.config.difficulty}")
    if gi.config.picked_topic:
        parts.append(f"Picked topic: {gi.config.picked_topic}")
    if gi.config.include_units:
        parts.append("Include units context (relevant snippets only).")
    if gi.config.include_subunits:
        parts.append("Consider subunits of the selected unit.")
    parts.append("Curriculum snippets (raw):\n" + curriculum_text)
    parts.append("Output a full exam in Markdown with tasks and sample solutions where appropriate.")
    return "\n\n".join(parts)


def render_validation_prompt(go: GenerationOutput, criteria_text: str, base_prompt: str | None = None) -> str:
    """Render the user content for validation.

    Args:
        go: Generation output to evaluate.
        criteria_text: Human-readable criteria description.
        base_prompt: Optional validator prefix.

    Returns:
        String to be used as user content in messages.create.
    """
    parts: List[str] = []
    if base_prompt:
        parts.append(base_prompt.strip())
    else:
        parts.append("Oceni sledeci generisani ispit po kriterijumima. Vrati JSON sa poljima: score [0..1], verdict ['pass'|'fail'], feedback (kratko).")
    parts.append("Kriterijumi:\n" + criteria_text)
    parts.append("Generisani ispit (Markdown):\n" + go.completion)
    parts.append("Odgovori samo JSON objektom bez dodatnog teksta.")
    return "\n\n".join(parts)


# -------- Batch helpers ---------

def build_generation_batch_items(inputs: List[GenerationInput], curricula: Dict[str, str], base_prompt: str | None = None, attachment_file_id: str | None = None) -> List[Dict[str, Any]]:
    """Create NDJSON-ready items for the generation stage.

    Each item is {"custom_id", "type": "message", "params": <messages payload>}.

    Example logical use:
        >>> items = build_generation_batch_items([gi], curricula)  # doctest: +SKIP
    """
    rows: List[Dict[str, Any]] = []
    for gi in inputs:
        curriculum_text = "\n\n".join(curricula.get(p, "") for p in gi.sample.curriculum_paths)
        prompt = render_generation_prompt(gi, curriculum_text, base_prompt)
        params = create_messages_payload(user_content=prompt, attachments=(
            [{"type": "file", "file_id": attachment_file_id}] if attachment_file_id else None
        ))
        rows.append({
            "custom_id": f"gen::{gi.sample.id}",
            "request": params,
        })
    return rows


def build_validation_batch_items(outputs: List[GenerationOutput], criteria_text: str, base_prompt: str | None = None, attachment_file_id: str | None = None) -> List[Dict[str, Any]]:
    """Create NDJSON-ready items for the validation stage.

    Example logical use:
        >>> items = build_validation_batch_items([go], "- criteria -")  # doctest: +SKIP
    """
    rows: List[Dict[str, Any]] = []
    for go in outputs:
        prompt = render_validation_prompt(go, criteria_text, base_prompt)
        params = create_messages_payload(user_content=prompt, attachments=(
            [{"type": "file", "file_id": attachment_file_id}] if attachment_file_id else None
        ))
        rows.append({
            "custom_id": f"val::{go.sample_id}",
            "request": params,
        })
    return rows


def parse_generation_batch_results(path: Path) -> List[GenerationOutput]:
    """Turn NDJSON responses into GenerationOutput objects."""
    rows = parse_ndjson(path)
    outputs: List[GenerationOutput] = []
    for r in rows:
        cid = r.get("custom_id", "")
        if not cid.startswith("gen::"):
            continue
        sample_id = cid.split("::", 1)[1]
        text = extract_text_from_response(r)
        outputs.append(GenerationOutput(sample_id=sample_id, prompt="", completion=text))
    return outputs


def parse_validation_batch_results(path: Path) -> List[ValidationOutput]:
    """Turn NDJSON responses into ValidationOutput objects, forgiving of non-JSON validator replies."""
    import json as _json
    rows = parse_ndjson(path)
    outs: List[ValidationOutput] = []
    for r in rows:
        cid = r.get("custom_id", "")
        if not cid.startswith("val::"):
            continue
        sample_id = cid.split("::", 1)[1]
        text = extract_text_from_response(r)
        try:
            data = _json.loads(text)
            outs.append(ValidationOutput(
                sample_id=sample_id,
                score=float(data.get("score", 0.0)),
                verdict=data.get("verdict", "fail"),
                feedback=str(data.get("feedback", "")),
            ))
        except Exception:
            outs.append(ValidationOutput(
                sample_id=sample_id,
                score=0.0,
                verdict="fail",
                feedback=f"Validator returned non-JSON: {text[:200]}...",
            ))
    return outs


# -------- Sequential (non-batch) one-off helpers ---------

def run_generation_sync(gi: GenerationInput, curricula: Dict[str, str], base_prompt: str | None = None, attachment_path: Path | None = None) -> GenerationOutput:
    """Synchronous convenience for generation.

    Example:
        >>> go = run_generation_sync(gi, curricula)  # doctest: +SKIP
    """
    curriculum_text = "\n\n".join(curricula.get(p, "") for p in gi.sample.curriculum_paths)
    prompt = render_generation_prompt(gi, curriculum_text, base_prompt)
    attachments = None
    if attachment_path and attachment_path.exists():
        file_id = upload_file(attachment_path)
        attachments = [{"type": "file", "file_id": file_id}]
    from .anthropic_utils import get_client, create_messages_payload
    client = get_client()
    resp = client.messages.create(**create_messages_payload(user_content=prompt, attachments=attachments))
    texts = []
    for c in resp.content:
        if c.type == "text":
            texts.append(c.text)
    completion = "\n".join(texts).strip()
    return GenerationOutput(sample_id=gi.sample.id, prompt=prompt, completion=completion)


def run_validation_sync(go: GenerationOutput, criteria_text: str, base_prompt: str | None = None, attachment_path: Path | None = None) -> ValidationOutput:
    """Synchronous convenience for validation.

    Example:
        >>> vo = run_validation_sync(go, "- criteria -")  # doctest: +SKIP
    """
    import json as _json
    prompt = render_validation_prompt(go, criteria_text, base_prompt)
    attachments = None
    if attachment_path and attachment_path.exists():
        file_id = upload_file(attachment_path)
        attachments = [{"type": "file", "file_id": file_id}]
    from .anthropic_utils import get_client, create_messages_payload
    client = get_client()
    resp = client.messages.create(**create_messages_payload(user_content=prompt, attachments=attachments))
    texts = []
    for c in resp.content:
        if c.type == "text":
            texts.append(c.text)
    result = "\n".join(texts).strip()
    try:
        data = _json.loads(result)
        return ValidationOutput(
            sample_id=go.sample_id,
            score=float(data.get("score", 0.0)),
            verdict=data.get("verdict", "fail"),
            feedback=str(data.get("feedback", "")),
        )
    except Exception:
        return ValidationOutput(
            sample_id=go.sample_id,
            score=0.0,
            verdict="fail",
            feedback=f"Validator returned non-JSON: {result[:200]}...",
        )
