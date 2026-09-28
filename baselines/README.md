# Baselines

Reference implementations of the comparison methods in Table 1. All of them read
the shared split in `data/splits/`, so their numbers are directly comparable to
the MASC detector's.

## `llm_as_detector/`

Prompts a judge LLM to attribute the failure directly. The prompts and evaluation
protocol are adopted **unmodified** from the official Who&When implementation, so
these numbers stay comparable to the ones published there.

| File | Role |
| --- | --- |
| `run_inference.py` | CLI entry point; picks a method, a model, and a subset |
| `prompt_methods.py` | The prompting strategies, driving Azure OpenAI deployments |
| `local_model.py` | The same strategies against local Llama / Qwen checkpoints |
| `evaluate.py` | Scores a prediction log: attribution accuracy or step-level metrics |
| `evaluate_step_level.py` | Stricter step-level protocol that penalises early flags |

Methods: `all_at_once`, `step_by_step`, `binary_search`, plus the `slide_window`
and `buffer` context-management variants.

Credentials come from `$AZURE_OPENAI_API_KEY` / `$AZURE_OPENAI_ENDPOINT` (or a
local `.env`). Never pass a key as a command-line argument — it lands in your
shell history and in process listings.

## `supervised/`

Trained *with* step-level error labels, on individual steps and without any
interaction history — the contrast MASC is measured against.

| File | Role |
| --- | --- |
| `train_bert.py` | Frozen `all-MiniLM-L6-v2` + trainable MLP head |
| `train_llm_classifier.py` | Frozen open-weight LLM + trainable classification head |

Both default to the per-subset hyperparameters in Table 6 of the paper; pass
`--epochs` / `--lr` / `--batch_size` to override.

See the [root README](../README.md#baselines) for runnable examples.
