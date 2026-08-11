@AGENTS.md

## Model-routing harness

The main Claude Code conversation runs on Fable and owns planning, architecture, ambiguous decisions, integration, and final verification.

Delegate routine, self-contained work proactively to the `opus-chores` subagent when doing so keeps disposable search results, logs, inventories, or mechanical work out of the main context. Good delegation targets include repository searches, file inventories, log triage, test execution, formatting, documentation checks, and tightly specified mechanical edits.

Do not delegate a tiny one-step action when the delegation overhead would exceed the context saved. Keep dependent work sequential; parallelize only independent tasks. Require concise evidence-based handoffs from the subagent, then let Fable make the final integration decision.
