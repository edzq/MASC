"""Anomaly-triggered self-correction (paper Sec. 3.5, Eq. 13).

MASC separates *deciding* to intervene from *performing* the correction:

* The gate below is the whole of Eq. 13's condition -- a step is revised only
  when its anomaly score exceeds the threshold ``delta``, which keeps
  interventions targeted and the overhead small.
* The correction itself is an LLM call made by the host multi-agent framework,
  because only the framework can re-run an agent and write the revised output
  back into its shared history. This module therefore supplies the prompt and
  the parsing of its reply, and leaves the call itself to the integration.

:data:`RECOVERY_PROMPT` is transcribed from the "Prompt for Response Recovery"
box in the paper's appendix.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

RECOVERY_PROMPT = """You are an AI agent playing the role of "{role}". \
You previously generated a response during a multi-agent reasoning process, \
but an anomaly detector flagged your output as potentially incorrect. Your \
task is to carefully reflect on whether your earlier response was indeed wrong \
given the original query and the current context.

Please follow these rules strictly:
1. Re-examine the original query and your earlier response in the context of your role.
2. If after reflection you believe your previous response is correct and does not \
require modification, explicitly state that no correction is needed.
3. If you identify errors or find a better answer, provide a corrected response.
4. Always output in the fixed JSON format below. Do not add extra explanations \
outside the JSON.

Output format:
{{
  "correction_needed": "Yes" or "No",
  "final_response": "If correction_needed=No, repeat your original response here. \
If Yes, provide the corrected response."
}}

Input Information:
- Query: {question}
- Your Previous Response: {response}
- Context (previous steps if available): {context}
"""


@dataclass
class CorrectionDecision:
    """Whether a step is revised, and what the reviser returned.

    Attributes:
        triggered: The gate fired, i.e. ``s(t) > delta``.
        score: The anomaly score that was compared against the threshold.
        output: The output to propagate downstream -- the original when the gate
            did not fire or the correction agent declined to change anything,
            and the revised text otherwise.
        corrected: The correction agent actually changed the output.
    """

    triggered: bool
    score: float
    output: str
    corrected: bool = False


def should_intervene(score: float, threshold: float) -> bool:
    """Eq. 13's gate: revise step ``t`` only when ``s(t) > delta``."""
    return score > threshold


def build_recovery_prompt(
    role: str,
    question: str,
    response: str,
    context: Sequence[str] = (),
    max_context_steps: Optional[int] = 2,
) -> str:
    """Render :data:`RECOVERY_PROMPT` for one flagged step.

    Args:
        role: Role description of the agent whose output was flagged.
        question: The task query.
        response: The flagged output that should be re-examined.
        context: Preceding steps, oldest first.
        max_context_steps: Keep only this many of the most recent steps. The
            paper's case study monitors two preceding steps; ``None`` keeps all.
    """
    steps: List[str] = list(context)
    if max_context_steps is not None:
        steps = steps[-max_context_steps:]
    rendered = "\n".join(f"- {step}" for step in steps) if steps else "(none)"
    return RECOVERY_PROMPT.format(
        role=role, question=question, response=response, context=rendered
    )


def parse_recovery_reply(reply: str, original: str) -> CorrectionDecision:
    """Parse a correction agent's JSON reply into a decision.

    Tolerates the usual LLM deviations -- fenced code blocks, prose around the
    object -- by extracting the first balanced-looking JSON object. If nothing
    parses, or the agent answered "No", the original output is kept: a
    correction step must never be able to corrupt a response it failed to
    understand.
    """
    payload = None
    match = re.search(r"\{.*\}", reply, re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            payload = None

    if not isinstance(payload, dict):
        return CorrectionDecision(triggered=True, score=float("nan"), output=original)

    needed = str(payload.get("correction_needed", "No")).strip().lower() == "yes"
    final = payload.get("final_response") or original
    return CorrectionDecision(
        triggered=True,
        score=float("nan"),
        output=final if needed else original,
        corrected=needed and final != original,
    )
