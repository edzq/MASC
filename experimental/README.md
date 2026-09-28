# Experimental

Exploratory code kept for provenance. **None of it is part of the paper's
reported results** — for the method see [`masc/`](../masc) and
[`scripts/`](../scripts).

## `learnable_step_selector.py`

An earlier attempt at making detection context-aware by learning *which* past
steps to attend to: a scorer ranks the buffered history against the current step,
the top-K are pooled into a context vector, and a supervised head classifies the
result.

It motivated the context-aware framing that MASC ended up addressing differently
— next-execution reconstruction conditions on the whole history and needs no
error labels, whereas this variant is supervised.

```bash
python experimental/learnable_step_selector.py --subset Algorithm-Generated
```
