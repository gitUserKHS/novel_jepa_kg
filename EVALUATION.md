# Evaluation Plan

Compare three generation modes:

1. LLM-only
2. RAG + LLM
3. JEPA Planner + RAG + LLM

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
- generation outputs from three modes
- metric table
- qualitative observations
- limitations
