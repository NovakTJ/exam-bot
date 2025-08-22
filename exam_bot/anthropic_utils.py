from __future__ import annotations
import json
import os
from typing import Iterable, List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm

from anthropic import Anthropic, AsyncAnthropic


DEFAULT_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-7-sonnet-20250219")


@dataclass
class BatchJob:
    id: str
    input_file: Path
    output_dir: Path


def get_client() -> Anthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    return Anthropic(api_key=api_key)


def create_messages_payload(
    user_content: str,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    max_tokens: int = 2048,
    temperature: float = 0.7,
    attachments: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model or DEFAULT_MODEL,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "user", "content": user_content},
        ],
        "temperature": temperature,
    }
    if system_prompt:
        payload["system"] = system_prompt
    if attachments:
        payload["attachments"] = attachments
    return payload


def write_batch_items(
    items: List[Dict[str, Any]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def prepare_batch_file(
    rows: Iterable[Dict[str, Any]],
    input_path: Path,
) -> None:
    items = []
    for i, row in enumerate(rows):
        custom_id = row.get("custom_id") or f"item-{i:05d}"
        payload = row["request"]
        items.append({
            "custom_id": custom_id,
            "params": {
                "type": "message",
                **payload,
            },
        })
    write_batch_items(items, input_path)


def upload_file(path: Path, purpose: str = "message") -> str:
    """Upload a file to Anthropic Files API and return file_id."""
    client = get_client()
    with path.open("rb") as f:
        file_obj = client.files.create(purpose=purpose, file=f)
    return file_obj.id


def run_batch(input_file: Path, output_dir: Path) -> BatchJob:
    client = get_client()
    output_dir.mkdir(parents=True, exist_ok=True)
    # Upload the NDJSON input via Files API
    with input_file.open("rb") as f:
        upload = client.files.create(purpose="batch", file=f)
    job = client.messages.batches.create(input_file_id=upload.id)
    return BatchJob(id=job.id, input_file=input_file, output_dir=output_dir)


def poll_batch(job: BatchJob, interval_seconds: float = 5.0) -> Dict[str, Any]:
    client = get_client()
    import time
    while True:
        b = client.messages.batches.retrieve(job.id)
        status = getattr(b, "processing_status", None) or getattr(b, "status", None)
        if status in {"ended", "failed", "cancelled", "completed", "succeeded"}:
            return b.model_dump() if hasattr(b, "model_dump") else b.__dict__
        time.sleep(interval_seconds)


def download_batch_results(job: BatchJob) -> Path:
    client = get_client()
    b = client.messages.batches.retrieve(job.id)
    out_path = job.output_dir / f"{job.id}.ndjson"
    # Prefer output_file_id if available
    output_file_id = getattr(b, "output_file_id", None)
    if output_file_id:
        # Stream the file content
        content = client.files.content(output_file_id)
        with out_path.open("wb") as f:
            for chunk in content.iter_bytes():
                f.write(chunk)
        return out_path
    # Fallback to results() stream if supported
    try:
        result = client.messages.batches.results(job.id)
        with out_path.open("wb") as f:
            for chunk in result.iter_bytes():
                f.write(chunk)
        return out_path
    except Exception:
        # As last resort, write error JSON
        out_path.write_text(json.dumps(b if isinstance(b, dict) else b.__dict__), encoding="utf-8")
        return out_path


def parse_ndjson(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def extract_text_from_response(resp: Dict[str, Any]) -> str:
    """Handle multiple result shapes:
    - {"type":"response","response":{"content":[{"type":"text","text":"..."}]}}
    - {"result":{"message":{"content":[{"type":"text","text":"..."}]}}}
    - {"error": ...}
    """
    # Common shapes
    try:
        if "response" in resp and isinstance(resp["response"], dict):
            content = resp["response"].get("content", [])
        elif "result" in resp and isinstance(resp["result"], dict):
            content = resp["result"].get("message", {}).get("content", [])
        else:
            content = resp.get("content", [])
        texts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
        return "\n".join(texts).strip() or json.dumps(resp)
    except Exception:
        return json.dumps(resp)


def call_messages(
    user_content: str,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    max_tokens: int = 2048,
    temperature: float = 0.7,
) -> str:
    client = get_client()
    resp = client.messages.create(
        **create_messages_payload(user_content, system_prompt, model, max_tokens, temperature)
    )
    # anthropic SDK returns content list
    texts = []
    for c in resp.content:
        if c.type == "text":
            texts.append(c.text)
    return "\n".join(texts).strip()
