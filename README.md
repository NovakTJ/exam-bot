# exam-bot

Minimal pipeline to generate and validate OOP exams using Anthropic Claude (supports batch API and synchronous runs).

## Setup

- Create and activate a Python 3.9+ environment
- Install deps:

```
pip install -r requirements.txt
```

- Create `.env` and set `ANTHROPIC_API_KEY` (see `.env.example`).

## Quick start

- One-shot generation (sync) — attaches `Primer ispita.pdf` by default:

```
python -m exam_bot.cli gen-sync --difficulty medium
```

- Full batch pipeline (generation -> validation) — attaches `Primer ispita.pdf` by default:

```
python -m exam_bot.cli run-batch-pipeline --n-samples 3
```

- Validate a saved Markdown exam (sync) — attaches `Primer ispita.pdf` by default:

```
python -m exam_bot.cli validate-sync exams/just_oneshot/1.md
```

To disable attachments, add `--no-attach` to the commands.

Outputs are stored under `./batches/` (inputs, results, summary.json).

## New: Send an arbitrary prompt (optionally with context)

Use the minimal CLI to send a prompt to Anthropic's Messages API, with optional context loaded from a file or passed inline:

```
python -m exam_bot.cli --prompt "Summarize the syllabus" --context-file "./Primer ispita.md"
```

Options:
- `--context-file` or `--context` to pass extra text
- `--model`, `--max-tokens`, `--temperature`

## Notes

- You can tweak difficulty and selection of curriculum snippets via CLI options.
- Extend `criteria_text` for stricter validation. Consider structuring it as JSON and adjusting the validator prompt accordingly.
- To try alternative prompting styles, update the `render_generation_prompt` and `render_validation_prompt` in `exam_bot/pipeline.py`.