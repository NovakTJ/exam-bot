from __future__ import annotations
"""Anthropic helpers: client setup, batching, file uploads, and small utilities.

This module wraps common Anthropic operations used by the pipeline:
- Creating a configured client (with optional Files API beta header via env).
- Preparing and submitting Messages Batches with NDJSON input.
- Polling for completion and downloading results.
- Optional file upload helper (SDK first, HTTP fallback).
- Simple sync messages call convenience.

Environment variables:
- ANTHROPIC_API_KEY: required.
- ANTHROPIC_MODEL: default model id (optional; defaults to claude-3-7-sonnet-20250219).
- ANTHROPIC_BETA: beta header value, defaults to files-api-2025-04-14.

Examples are marked as doctest +SKIP because they require network and a valid API key.
"""

import json
import os
from typing import Iterable, List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm

from anthropic import Anthropic, AsyncAnthropic
import httpx


DEFAULT_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-7-sonnet-20250219")


@dataclass
class BatchJob:
    """Lightweight handle for a Messages Batch job.

    Attributes:
        id: Server-side batch job id returned by Anthropic.
        input_file: Local NDJSON file that was uploaded for this batch.
        output_dir: Directory where result NDJSON will be saved.

    Example:
        >>> from pathlib import Path
        >>> BatchJob(id="batch_123", input_file=Path("in.ndjson"), output_dir=Path("batches"))
        BatchJob(id='batch_123', input_file=PosixPath('in.ndjson'), output_dir=PosixPath('batches'))
    """
    id: str
    input_file: Path
    output_dir: Path


def get_client() -> Anthropic:
    """Create and return a configured Anthropic client.

    Respects env vars for API key and beta header (for Files API support).

    Returns:
        Anthropic client instance.

    Raises:
        RuntimeError: if ANTHROPIC_API_KEY is not set.

    Example:
        >>> client = get_client()  # doctest: +SKIP
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    # Enable Files API via beta header; allow override/merge via env var
    beta = os.getenv("ANTHROPIC_BETA", "files-api-2025-04-14")
    default_headers = {"anthropic-beta": beta}
    return Anthropic(api_key=api_key, default_headers=default_headers)


def _api_headers() -> Dict[str, str]:
    """Build HTTP headers for raw API calls.

    Includes: x-api-key, anthropic-version, anthropic-beta.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    beta = os.getenv("ANTHROPIC_BETA", "files-api-2025-04-14")
    # anthropic-version remains required for HTTP calls
    return {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "anthropic-beta": beta,
    }


def create_messages_payload(
    user_content: str,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    max_tokens: int = 2048,
    temperature: float = 0.7,
    attachments: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build a messages.create payload for Anthropic.

    Args:
        user_content: Main textual content to send as the user message.
        system_prompt: Optional system message.
        model: Model id; falls back to DEFAULT_MODEL.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        attachments: Optional attachments, e.g. [{"type": "file", "file_id": "..."}].

    Returns:
        Dict suitable for client.messages.create(**payload).

    Example:
        >>> payload = create_messages_payload("Hello")
        >>> payload["messages"][0]["content"][0]["text"]
        'Hello'
    """
    payload: Dict[str, Any] = {
        "model": model or DEFAULT_MODEL,
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_content},
                ],
            },
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
    """Write NDJSON items to a file.

    Each item should already be a dict with keys like
    {"custom_id": "...", "type": "message", "params": { ... }}.

    Example:
        >>> from pathlib import Path
        >>> write_batch_items([{ "custom_id": "x", "type": "message", "params": {"model": "m", "messages": []}}], Path("/tmp/x.ndjson"))  # doctest: +SKIP
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def prepare_batch_file(
    rows: Iterable[Dict[str, Any]],
    input_path: Path,
) -> None:
    """Convert logical rows to Messages Batch NDJSON format and save.

    Input rows should be of the form: {"custom_id": str, "request": <messages payload>}.

    Example:
        >>> rows = [{"custom_id": "a", "request": {"model": "m", "messages": []}}]
        >>> from pathlib import Path
        >>> prepare_batch_file(rows, Path("/tmp/batch.ndjson"))  # doctest: +SKIP
    """
    items = []
    for i, row in enumerate(rows):
        custom_id = row.get("custom_id") or f"item-{i:05d}"
        payload = row["request"]
        items.append({
            "custom_id": custom_id,
            "type": "message",
            "params": payload,
        })
    write_batch_items(items, input_path)


def upload_file(path: Path, purpose: str = "message") -> str:
    """Upload a file to Anthropic Files API and return the file_id.

    Tries the SDK first, then falls back to HTTP with beta header.

    Args:
        path: Local file path to upload.
        purpose: "message" for attachments, or "batch" for NDJSON batch inputs.

    Returns:
        The uploaded file id.

    Example:
        >>> fid = upload_file(Path("Primer ispita.pdf"))  # doctest: +SKIP
    """
    client = get_client()
    # Try SDK first
    try:
        with path.open("rb") as f:
            file_obj = getattr(client, "files").create(purpose=purpose, file=f)  # may raise
        return file_obj.id
    except Exception:
        # Fall back to raw HTTP
        url = "https://api.anthropic.com/v1/files"
        headers = _api_headers()
        with path.open("rb") as f:
            files = {"file": (path.name, f, "application/octet-stream")}
            data = {"purpose": purpose}
            with httpx.Client(timeout=60) as http:
                resp = http.post(url, headers=headers, files=files, data=data)
                resp.raise_for_status()
                return resp.json()["id"]


def run_batch(input_file: Path, output_dir: Path) -> BatchJob:
    """Submit a Messages Batch from a prepared NDJSON file.

    Attempts SDK Files API first (with completion_window), then HTTP fallback.

    Args:
        input_file: NDJSON file created by prepare_batch_file.
        output_dir: Where result NDJSON will be saved.

    Returns:
        BatchJob containing the server id and locations.

    Example:
        >>> job = run_batch(Path("batches/gen_inputs.jsonl"), Path("batches/gen_results"))  # doctest: +SKIP
    """
    client = get_client()
    output_dir.mkdir(parents=True, exist_ok=True)
    # Prefer Files API; if unavailable, fall back to passing the path directly
    try:
        with input_file.open("rb") as f:
            upload = getattr(client, "files").create(purpose="batch", file=f)  # may raise
        job = client.messages.batches.create(input_file_id=upload.id, completion_window="24h")
    except Exception:
        # Raw HTTP fallback
        file_id = upload_file(input_file, purpose="batch")
        url = "https://api.anthropic.com/v1/messages/batches"
        headers = _api_headers() | {"content-type": "application/json"}
        payload = {"input_file_id": file_id, "completion_window": "24h"}
        with httpx.Client(timeout=60) as http:
            resp = http.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            job = resp.json()
            return BatchJob(id=job["id"], input_file=input_file, output_dir=output_dir)
    return BatchJob(id=job.id, input_file=input_file, output_dir=output_dir)


def poll_batch(job: BatchJob, interval_seconds: float = 5.0) -> Dict[str, Any]:
    """Poll the batch job until a terminal status and return the final object.

    Terminal statuses include: ended, failed, cancelled, completed, succeeded.

    Example:
        >>> status = poll_batch(job)  # doctest: +SKIP
    """
    client = get_client()
    import time
    while True:
        try:
            b = client.messages.batches.retrieve(job.id)
            status = getattr(b, "processing_status", None) or getattr(b, "status", None)
            if status in {"ended", "failed", "cancelled", "completed", "succeeded"}:
                return b.model_dump() if hasattr(b, "model_dump") else b.__dict__
        except Exception:
            # HTTP fallback
            url = f"https://api.anthropic.com/v1/messages/batches/{job.id}"
            headers = _api_headers()
            with httpx.Client(timeout=30) as http:
                resp = http.get(url, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                status = data.get("processing_status") or data.get("status")
                if status in {"ended", "failed", "cancelled", "completed", "succeeded"}:
                    return data
        time.sleep(interval_seconds)


def download_batch_results(job: BatchJob) -> Path:
    """Download completed batch results to an NDJSON file and return its path.

    Tries the SDK streaming endpoint first, then falls back to HTTP.

    Example:
        >>> out = download_batch_results(job)  # doctest: +SKIP
    """
    client = get_client()
    try:
        b = client.messages.batches.retrieve(job.id)
    except Exception:
        # HTTP fallback to retrieve batch info
        url = f"https://api.anthropic.com/v1/messages/batches/{job.id}"
        headers = _api_headers()
        with httpx.Client(timeout=30) as http:
            resp = http.get(url, headers=headers)
            resp.raise_for_status()
            b = resp.json()
    out_path = job.output_dir / f"{job.id}.ndjson"
    # Prefer output_file_id if available
    output_file_id = getattr(b, "output_file_id", None) if not isinstance(b, dict) else b.get("output_file_id")
    if output_file_id:
        try:
            content = client.files.content(output_file_id)
            with out_path.open("wb") as f:
                for chunk in content.iter_bytes():
                    f.write(chunk)
            return out_path
        except Exception:
            # HTTP fallback
            url = f"https://api.anthropic.com/v1/files/{output_file_id}/content"
            headers = _api_headers()
            with httpx.Client(timeout=None) as http:
                with http.stream("GET", url, headers=headers) as r:
                    r.raise_for_status()
                    with out_path.open("wb") as f:
                        for chunk in r.iter_bytes():
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
        out_path.write_text(json.dumps(b if isinstance(b, dict) else getattr(b, "__dict__", {})), encoding="utf-8")
        return out_path


def parse_ndjson(path: Path) -> List[Dict[str, Any]]:
    """Read an NDJSON file into a list of dicts.

    Example:
        >>> rows = parse_ndjson(Path("batches/any.ndjson"))  # doctest: +SKIP
    """
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
    """Send a single messages.create call and return concatenated text.

    Example:
        >>> call_messages("Say hi")  # doctest: +SKIP
        'Hi!'
    """
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
