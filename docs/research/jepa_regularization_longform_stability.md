# JEPA Regularization and Long-form Stability Review

Date: 2026-07-19

## Executive decision

This project should not apply exact SIGReg or VISReg directly to its frozen,
L2-normalized text targets. Instead, it uses a target-distribution-relative,
VISReg-inspired regularizer on predictor outputs and keeps the original
predicted-target alignment loss.

The long-form generator uses a resource-bounded hierarchy:

```text
global story outline
-> JEPA next-state direction
-> memory/KG/RAG context
-> one section draft
-> local repetition and stability gate
-> at most one combined revision
-> atomic draft, memory, ledger, outline, and state persistence
```

## Why anti-collapse regularization matters

- VICReg prevents constant representations with explicit variance and
  covariance terms.
- LeJEPA proposes SIGReg, matching learned embeddings to an isotropic Gaussian
  through random sketches with linear time and memory complexity.
- VISReg separates scale from distributional shape. Its variance term maintains
  scale, while a sliced-Wasserstein sketching term preserves fuller distribution
  shape and supplies stronger gradients under collapse.

These methods were developed for learned representation spaces. Novel JEPA Lab
has a different geometry: the embedding model is frozen and returns
L2-normalized text vectors, while only the next-state predictor is trainable.

## Project-specific geometry

The current artifact contains 10 current/next pairs in 768 dimensions.

- Every target has norm close to 1 because the text embeddings are normalized.
- Mean target standard deviation per dimension is about 0.0164.
- Ten centered samples have rank at most 9, regardless of the 768-dimensional
  ambient space.
- Forcing unit variance in every dimension would move predictions away from the
  actual target manifold.
- A covariance or Gaussian-shape claim cannot be supported by ten samples.

Therefore exact isotropic-Gaussian regularization is disabled for this artifact.
The current checkpoint remains a research artifact and cannot pass the consumer
service gate.

## Implemented regularizer

`target_visreg_loss` combines three target-relative terms:

1. Scale floor: predictor standard deviation must retain a configurable fraction
   of target-batch standard deviation.
2. Shape matching: random one-dimensional projections of predictions and targets
   are sorted and compared as a sliced-Wasserstein approximation.
3. Center matching: predictor and target batch centers are aligned after scaling
   by target standard deviation.

The existing cosine and normalized-MSE next-state losses remain the invariance or
prediction-alignment objective. The distribution term is weighted separately.

Default operating rules:

- Method: `target_visreg`
- Weight: `0.03`
- Minimum dataset size: `40`
- Minimum active batch size: `8`
- Random projections: `64`
- Scale floor ratio: `0.75`
- Consumer quality gate: predicted/target normalized effective-rank ratio `>= 0.50`
- Recommended research dataset: at least `96` transitions

The model card and checkpoint record whether the regularizer was active, why it
was skipped, output and target effective ranks, their ratio, and full-dataset
predicted-target cosine.

## Long-form stability design

The local `gemma4:e4b` process remains within the 8K context and 8GB VRAM budget.
The system does not feed the full manuscript back on every turn.

### Global planning

A 6-20 beat outline stores each beat's act, narrative purpose, required state
change, setup/payoff, and forbidden repeat. It is generated once per story and
then persisted. Every section receives its active beat, previous consequence,
and next setup constraint.

### Working context

Each section receives only:

- the recent prose window;
- the global active outline beat;
- latest entity states and relevant KG relations;
- retrieved section memories and unresolved clues;
- consumed narrative beats;
- one primary narrative function.

This keeps stable canon close to the generation instruction without exhausting
the model context with old prose.

### Post-generation gate

The local gate checks:

- minimum body length and complete sentence ending;
- duplicate subtitles and consumed narrative beats;
- unknown or mutated character names;
- entity-state changes without a local cause;
- reopening an already resolved clue without a new cause.

Repetition and stability findings are combined into one revision prompt. A
section receives at most one revision, and the revised candidate replaces the
draft only when it reduces the issue count or repairs a hard truncation. This is
important on the target GPU: several independent critic passes would increase
latency and Ollama load failures.

### Persistence and observability

After every accepted section, the service atomically persists the draft,
structured memory, latest-state/KG ledger, global outline, and run state. Worker
metrics include section stability score, issue count, hard-failure flag,
revision count/success, memory retrievals, repetition, progression, controlled
creativity, and JEPA retrieval score.

## Evaluation protocol

For a defensible capstone experiment:

1. Collect at least 96 diverse transitions and keep a fixed validation split of
   at least 6 samples.
2. Compare the same predictor architecture with regularizer `none` and
   `target_visreg` over at least five random seeds.
3. Report validation cosine, hit@5, top-1 diversity, effective-rank ratio, JEPA
   gain over RAG-next, and mean/variance across seeds.
4. Generate matched 30,000-character stories with identical prompts and
   creativity profiles.
5. Plot consistency errors by relative story position, especially the middle
   40-70 percent where long-form systems often degrade.
6. Report latency, revision rate, Ollama failures, and accepted characters per
   minute together with quality metrics.

## Sources

- Bardes, Ponce, and LeCun. [VICReg](https://arxiv.org/abs/2105.04906), ICLR 2022.
- Balestriero and LeCun. [LeJEPA and SIGReg](https://arxiv.org/abs/2511.08544), 2025.
- Wu, Balestriero, and Levine. [VISReg](https://arxiv.org/abs/2606.02572), 2026.
- Yang et al. [Re3](https://aclanthology.org/2022.emnlp-main.296/), EMNLP 2022.
- Yang et al. [DOC](https://aclanthology.org/2023.acl-long.190/), ACL 2023.
- Wang et al. [DOME](https://aclanthology.org/2025.naacl-long.63/), NAACL 2025.
- Li et al. [Lost in Stories / ConStory-Bench](https://arxiv.org/abs/2603.05890), 2026.

## Scope warning

This remains a JEPA-inspired narrative transition predictor, not a reproduction
of image JEPA, LeJEPA, SIGReg, or VISReg. The implemented loss borrows VISReg's
scale-and-shape separation while matching the project's fixed target embedding
distribution.
