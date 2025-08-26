from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from pydantic import BaseModel
from dotenv import load_dotenv

from .clients.anthropic_client import AnthropicTextClient, AnthropicTextConfig

app = typer.Typer(help="Exam Bot CLI")


class RunArgs(BaseModel):
    prompt: Optional[str] = None
    prompt_file: Optional[Path] = None
    context_file: Optional[Path] = None
    context: Optional[str] = None
    model: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    repeat: int = 1
    output_dir: Optional[Path] = None


@app.command()
def run(prompt: Optional[str] = typer.Option(None, "--prompt", "-p"),
        prompt_file: Optional[Path] = typer.Option(None, "--prompt-file", "-P", exists=True, readable=True),
        context_file: Optional[Path] = typer.Option(None, "--context-file", "-c", exists=True, readable=True),
        context: Optional[str] = typer.Option(None, "--context", "-x"),
        model: Optional[str] = typer.Option(None, "--model", "-m"),
        max_tokens: Optional[int] = typer.Option(None, "--max-tokens"),
        temperature: Optional[float] = typer.Option(None, "--temperature"),
        repeat: int = typer.Option(1, "--repeat", "-r", min=1),
        output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o")):
    """Send a prompt (and optional context) to Anthropic Messages API and print the result.

    You can pass the prompt directly or via --prompt-file. Use --repeat to send multiple times.
    """
    load_dotenv()
    # Load prompt
    prompt_text = None
    if prompt_file is not None:
        prompt_text = prompt_file.read_text(encoding="utf-8")
    elif prompt is not None:
        prompt_text = prompt
    if not prompt_text:
        typer.secho("Provide --prompt or --prompt-file", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)

    # Load context
    ctx_text = None
    if context_file is not None:
        ctx_text = context_file.read_text(encoding="utf-8")
    elif context is not None:
        ctx_text = context

    client = AnthropicTextClient(AnthropicTextConfig())
    # Prepare output dir
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(1, repeat + 1):
        msg = client.run(prompt_text, ctx_text, model=model, max_tokens=max_tokens, temperature=temperature)
        if not msg.content:
            typer.secho(f"Run {i}: No output returned.", fg=typer.colors.YELLOW)
        else:
            # Print to stdout
            if repeat > 1:
                typer.echo(f"\n===== Run {i} =====\n")
            typer.echo(msg.content)
            # Write to file if requested
            if output_dir is not None:
                out_path = output_dir / f"result_{i:03d}.md"
                out_path.write_text(msg.content, encoding="utf-8")


if __name__ == "__main__":
    app()
