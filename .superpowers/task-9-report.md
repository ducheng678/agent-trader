# Task 9: coordinator, specialists, objective reflection

Implementation commit: `b39681c9e0e597fe3e7cda19505ef596de27bad0`.
Whitespace follow-up: `422e50b`.

## Files and APIs

- `market_agent/workflow_coordinator_agent.py`: `plan_request(request, budget, task_types=None)` returns a fixed-catalog `CoordinatorPlan`; `bind_contexts(plan, contexts)` pins actual summary IDs; `dispatch_tasks(plan, contexts, driver, grants, deadline_epoch=None, authorize=None)` returns `DispatchSpec` tuples when driver is `None`, invokes a host dispatcher callable with each spec, or calls an injected `AgentDriver` with mandatory host authorization. It never issues grants. `reconcile_reports(plan, reports, budget)` returns a typed routing directive for continue/wait/reschedule/reconciliation/safe unknown. `reschedule(plan, directive, budget)` creates a bounded new revision and fresh task IDs, requiring the host to issue new grants. `summarize_result(plan, reports, decision=None, risk=None, informational_answer=None, route_history=())` returns strict `WorkflowResult`; trading results require supplied risk approval, and missing evidence/errors yield unknown/no_trade.
- `market_agent/workflow_agents/`: six focused modules (`market_context_agent`, `event_filter_agent`, `fundamental_agent`, `technical_agent`, `decision_planner_agent`, `escalation_agent`) each expose `PROMPT_PROFILE_ID`, `build_messages`, `build_invocation`, and `run_node`. Their tasks and contexts must match the fixed profile, trace, workflow, task, summary ID, schema, prompt version, and three-to-five analysis steps. No tools are directly exposed; these modules only construct requests and use the injected driver.
- `market_agent/workflow_agents/common.py`: fixed immutable profile values, `prompt_release_registry()`, `output_schemas()`, `profile_for`, `checked_context`, and common request/report adapters. Driver-facing output is `SpecialistOutput[T]` with `conclusion`, typed `result`, and `evidence_refs`. The unknown envelope has conclusion `不知道` and null result. `JsonContractSchema` preserves strict JSON enum semantics while adapting existing specialist contracts to AgentDriver's fixed-abstention interface. The report's summary contains the canonical typed result JSON for downstream parsing. Reconciliation uses the escalation result schema with its own fixed profile.
- `market_agent/workflow_reflection_agent.py`: `reflect_output(target, target_kind, context, output_model=None, reviewer=None)` accepts only the three configured core kinds and emits `ReflectionResult`. The reviewer returns `ObjectiveReview` with exactly six allowlisted checks; it cannot select disposition. Missing or failed review fails closed. `reflection_output_schema()`, `reflection_release()`, and `run_reflection(request, driver, deadline_epoch, cost_limit_usd, grant, authorize)` provide a Luna-only, one-attempt, no-tool driver path. `build_correction_context`, `apply_correction_patch`, and `correct_output` support one allowlisted existing-field patch followed by at most one full rewrite. Replacements must improve the objective error tuple, preserve evidence and direction, avoid cycles, and not widen stop distance.

## Host integration

Configure AgentDriver with the specialist release registry and schemas; add the reflection release/schema when reflection runs through that driver. The host dispatcher/authorizer verifies grants and records dispatch policy/audit. AgentDriver owns model/retry/cost execution. Contexts are a map keyed by task ID and contain strict ContextSummary or ContextHandoff values. Grant values stay with host dispatch and never enter model payloads.

The graph must call reflection for core outputs before admitting them downstream; reflection functions do not mutate shared state. Correction generator/reviewer callables must use the host's existing shared time/attempt/cost limits. `correct_output` bounds orchestration to one patch and one rewrite, but does not own a second budget ledger.

## Lightweight validation

- Focused compileall succeeded.
- Constructed all seven specialist prompt releases and all seven strict output schemas; constructed reflection release/schema successfully.
- Exercised request → three catalogued tasks → three strict summarized handoffs → bounded DispatchSpecs → two messages each; each task had three analysis steps and missing reports returned `wait`.
- Staged whitespace check passed after the small line-ending follow-up.
- No broad test run, following the user's implementation-first direction.
- Did not modify `workflow_capabilities.py`, agents, `.tmpbudget`, or the concurrently built graph.
