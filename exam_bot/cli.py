from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict

import typer
from dotenv import load_dotenv
from rich import print
from tqdm import tqdm

from .types import GenerationInput, GenerationConfig, Sample
from .pipeline import (
    build_generation_batch_items,
    build_validation_batch_items,
    parse_generation_batch_results,
    parse_validation_batch_results,
    run_batch,
    poll_batch,
    download_batch_results,
    run_generation_sync,
    run_validation_sync,
)

load_dotenv()
app = typer.Typer(add_completion=False)


@app.command()
def gen_sync(
    sample_id: str = typer.Option("sample-001", help="ID for the sample"),
    curriculum_glob: str = typer.Option("data/page_*_extracted_text.txt", help="Glob for curriculum files to include"),
    difficulty: str = typer.Option("medium", help="Difficulty level"),
    include_units: bool = typer.Option(False, help="Include units context"),
    include_subunits: bool = typer.Option(False, help="Consider subunits of selected unit"),
    picked_topic: str = typer.Option("", help="Picked topic focus"),
    gen_prompt_file: Path = typer.Option(Path("exam_bot/prompts/generation_one_shot.txt"), help="Base generation prompt file"),
    attach_file: Path = typer.Option(Path("Primer ispita.pdf"), help="File (e.g., PDF) to attach to the prompt"),
    no_attach: bool = typer.Option(False, help="Do not attach any file"),
):
    """Run a single synchronous generation and print Markdown.

    Examples:
        python -m exam_bot.cli gen-sync --difficulty medium
        python -m exam_bot.cli gen-sync --no-attach
    """
    """Run a single synchronous generation and print Markdown."""
    paths = sorted(Path(".").glob(curriculum_glob))[:5]
    curricula: Dict[str, str] = {str(p): p.read_text(encoding="utf-8") for p in paths}

    gi = GenerationInput(
        sample=Sample(id=sample_id, curriculum_paths=[str(p) for p in paths]),
        config=GenerationConfig(
            prompt_template="one-shot",
            difficulty=difficulty,
            include_units=include_units,
            include_subunits=include_subunits,
            picked_topic=picked_topic or None,
        ),
    )
    base_prompt = gen_prompt_file.read_text(encoding="utf-8") if gen_prompt_file.exists() else None
    effective_attach = None if no_attach else attach_file
    go = run_generation_sync(gi, curricula, base_prompt, effective_attach)
    print(go.completion)


@app.command()
def run_batch_pipeline(
    gen_list_path: Path = typer.Option(Path("./batches/gen_inputs.jsonl"), help="Where to save gen batch input"),
    gen_out_dir: Path = typer.Option(Path("./batches/gen_results"), help="Where to save gen results"),
    val_list_path: Path = typer.Option(Path("./batches/val_inputs.jsonl"), help="Where to save val batch input"),
    val_out_dir: Path = typer.Option(Path("./batches/val_results"), help="Where to save val results"),
    n_samples: int = typer.Option(3, help="How many samples to generate"),
    curriculum_glob: str = typer.Option("data/page_*_extracted_text.txt"),
    criteria_text: str = typer.Option("- pokrivenost tema; - jezik srpski; - razumna duzina"),
    difficulty: str = typer.Option("medium", help="Difficulty level"),
    include_units: bool = typer.Option(False),
    include_subunits: bool = typer.Option(False),
    picked_topic: str = typer.Option(""),
    gen_prompt_file: Path = typer.Option(Path("exam_bot/prompts/generation_one_shot.txt")),
    val_prompt_file: Path = typer.Option(Path("exam_bot/prompts/validation_criteria_basic.txt")),
    attach_file: Path = typer.Option(Path("Primer ispita.pdf"), help="File to attach to both stages (uploaded once)"),
    no_attach: bool = typer.Option(False, help="Do not attach any file"),
):
    """Two-stage pipeline over batches: generation, then validation of those results.

    This writes NDJSON inputs/outputs to the batches/ folder and a summary.json.

    Example:
        python -m exam_bot.cli run-batch-pipeline --n-samples 3
    """
    """Two-stage pipeline over batches: generation, then validation of those results."""
    # 1) Build small set of samples
    paths = sorted(Path(".").glob(curriculum_glob))[:5]
    curricula: Dict[str, str] = {str(p): p.read_text(encoding="utf-8") for p in paths}

    gen_inputs: List[GenerationInput] = []
    for i in range(n_samples):
        gen_inputs.append(
            GenerationInput(
                sample=Sample(id=f"s{i:03d}", curriculum_paths=[str(p) for p in paths]),
                config=GenerationConfig(
                    prompt_template="one-shot",
                    difficulty=difficulty,
                    include_units=include_units,
                    include_subunits=include_subunits,
                    picked_topic=picked_topic or None,
                ),
            )
        )

    # 2) Prepare and submit generation batch
    from .anthropic_utils import prepare_batch_file
    base_gen = gen_prompt_file.read_text(encoding="utf-8") if gen_prompt_file.exists() else None
    # Optional single upload for both stages
    attachment_file_id = None
    if (not no_attach) and attach_file and attach_file.exists():
        from .anthropic_utils import upload_file
        attachment_file_id = upload_file(attach_file)
    gen_items = build_generation_batch_items(gen_inputs, curricula, base_gen, attachment_file_id)
    prepare_batch_file(gen_items, gen_list_path)
    job = run_batch(gen_list_path, gen_out_dir)
    print(f"[bold]Submitted generation batch[/bold]: {job.id}")
    status = poll_batch(job)
    print(f"Generation batch status: {status.get('processing_status')}")
    gen_results_path = download_batch_results(job)

    # 3) Parse generation results
    gen_outputs = parse_generation_batch_results(gen_results_path)
    print(f"Parsed {len(gen_outputs)} generation results")

    # 4) Build and submit validation batch
    base_val = val_prompt_file.read_text(encoding="utf-8") if val_prompt_file.exists() else None
    val_items = build_validation_batch_items(gen_outputs, criteria_text, base_val, attachment_file_id)
    prepare_batch_file(val_items, val_list_path)
    vjob = run_batch(val_list_path, val_out_dir)
    print(f"[bold]Submitted validation batch[/bold]: {vjob.id}")
    vstatus = poll_batch(vjob)
    print(f"Validation batch status: {vstatus.get('processing_status')}")
    val_results_path = download_batch_results(vjob)

    # 5) Parse validation results and print summary
    val_outputs = parse_validation_batch_results(val_results_path)
    passed = sum(1 for v in val_outputs if v.verdict == "pass")
    print(f"Validation: {passed}/{len(val_outputs)} passed")
    summary = [v.model_dump() for v in val_outputs]
    out_json = Path("./batches/summary.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved summary to {out_json}")


@app.command()
def validate_sync(
    md_path: Path = typer.Argument(..., help="Path to generated exam markdown"),
    criteria_text: str = typer.Option("- pokrivenost tema; - jezik srpski; - razumna duzina"),
    val_prompt_file: Path = typer.Option(Path("exam_bot/prompts/validation_criteria_basic.txt"), help="Base validator prompt"),
    attach_file: Path = typer.Option(Path("Primer ispita.pdf"), help="File (e.g., PDF) to attach to the validator"),
    no_attach: bool = typer.Option(False, help="Do not attach any file"),
):
    """Validate a single Markdown exam via messages API (not batch).

    Example:
        python -m exam_bot.cli validate_sync exams/just_oneshot/1.md
    """
    """Validate a single Markdown exam via messages API (not batch)."""
    from .types import GenerationOutput
    go = GenerationOutput(sample_id=md_path.stem, prompt="", completion=md_path.read_text(encoding="utf-8"))
    base_val = val_prompt_file.read_text(encoding="utf-8") if val_prompt_file.exists() else None
    effective_attach = None if no_attach else attach_file
    vo = run_validation_sync(go, criteria_text, base_val, effective_attach)
    print(vo.model_dump())


if __name__ == "__main__":
    app()
