# Evaluation Plan

The automatic pipeline evaluates one planner-backed generation mode:

1. Creative Hallucination + JEPA

The controlled hallucination mode treats hallucination as creative expansion:
the model is asked to add plausible new details, clues, symbols, or emotional
inferences while preserving world rules, known character names, and JEPA/RAG
direction.

Long-form continuity is supported by story-memory RAG. Every completed section
stores a compact record of established facts, character and relationship state,
locations, unresolved or resolved clues, and important state changes. Before the
next section is generated, only the most relevant records are retrieved within a
bounded context budget.

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
- the approximately 30,000-character controlled hallucination novel
- controlled hallucination output and metrics
- section structure metrics
- story-memory ledger path and retrieval diagnostics
- metric table
- qualitative observations
- limitations
