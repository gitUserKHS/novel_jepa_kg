# Evaluation Plan

Compare four generation modes:

1. LLM-only
2. RAG + LLM
3. JEPA Planner + RAG + LLM
4. Controlled Hallucination + JEPA

The fourth mode treats hallucination as controlled creative expansion: the model
is asked to add plausible new details, clues, symbols, or emotional inferences
while preserving world rules, known character names, and JEPA/RAG direction.

## Automatic Metrics

### Repetition Rate
Detect repeated n-grams and repeated sentence patterns.

### Embedding Continuity Score
Embed previous scene and generated scene. Higher similarity means smoother scene transition, but too high may indicate no plot progression.

### Keyword Consistency Score
Track important character names, goals, locations, and rules.

### Contradiction Checklist
Rule-based checks for obvious contradictions:
- character name mismatch
- goal suddenly disappears
- location changes without transition
- state tags conflict
- scene repeats previous event without progress

### Controlled Hallucination Metrics
Separate useful creative additions from risky drift.

- Creative Expansion Rate: generated content tokens not present in the previous scene.
- Hallucination Presence: low, moderate, or high expansion signal.
- Target Alignment: closeness to the configured creative expansion target.
- Useful Hallucination Score: novel material weighted by consistency and progression.
- Hallucination Risk: novelty that is not balanced by grounding, plus contradiction penalties.

### Section Structure Metrics
Check whether longer prose is actually organized like a sectioned novel scene.

- Section Count Fit: closeness to the configured number of titled sections.
- Section Body Coverage: share of sections that have enough body text.
- Average Section Body Chars: average amount of prose under each subtitle.

## Human or LLM Judge Criteria

Score from 1 to 5:
- continuity
- plot progression
- emotional consistency
- setting consistency
- readability

## Report Output

Save a Markdown report containing:
- config snapshot
- dataset size
- training curve
- generation outputs from four modes
- controlled hallucination output and metrics
- section structure metrics
- metric table
- qualitative observations
- limitations
