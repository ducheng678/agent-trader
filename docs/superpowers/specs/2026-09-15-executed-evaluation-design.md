# Executed evaluation repair (B14)

The existing signed dataset and scoring rules remain valid, but a recorded observation
does not measure the current workflow. Keep recorded scoring as offline fixture
regression; add an explicit executed mode and label every artifact accordingly.

Executed mode iterates immutable dataset cases and supplies only `task_input` and
`trace_id` to a workflow executor. Expectations and recorded output must never enter
the executor context. The executor runs the current production workflow through a
provided application boundary and returns actual output, elapsed time, input/output
tokens, cost, violations and the same trace identity as a RecordedObservation.
Timeout, cancellation and cost policy remain those of the workflow itself. A case
failure becomes a scored invalid/unknown observation with a hard violation, not a
substitution of the historical recording.

`EvaluationRun` binds execution_mode (`recorded` or `executed`) and the precise
code/prompt/model-policy versions. A release-quality gate for the current system
requires executed mode; recorded mode can still produce an artifact but cannot claim
release success. Baseline comparison requires the same mode and dataset binding.
The CLI selects mode explicitly; executed mode requires a configured executor factory
and refuses a missing factory. No default network/provider invocation is hidden in a
read-only evaluation command.

Tests use a fake host application counting actual run_workflow calls and recording
different outputs from the fixture. Verify expectations never leak to application,
all cases run once with exact trace IDs, metrics derive from actual usage, recorded
artifacts remain labeled, and release gate rejects recorded mode. Current environment
has no live provider exercise; actual production success rate remains unmeasured until
the executed gate runs against configured infrastructure.

## Version-binding closure (evaluator v3)

The explicit host factory must return an executor with a host-owned
`attested_binding` (code revision, model-policy digest and expected prompt-bundle
digest). CLI flags are only the requested identity: the runner compares them with
the host identity before work and around every case. The production application
accepts an explicit `evaluation_code_revision` from its host build authority; its
`evaluation_binding` property derives the prompt-bundle digest from the current
prompt authority and the model-policy digest from its actual model/embedding
settings and pricing version. Without a host revision it refuses executed
evaluation. The workflow adapter reads that property and returns the actual
`WorkflowExecution.prompt_release_digest` with each case observation. A mismatch
aborts the run without a release artifact; a failed case without an observed prompt
can be scored invalid, but marks the whole run `binding_verified=false`, so the
release gate always denies it. Recorded and legacy v1/v2 artifacts cannot claim a
verified binding. This is host attestation, not cryptographic proof of a build or
deployment; a production factory must source code/model identity from its trusted
build/configuration authority, not copy CLI values. A host may pass the revision
through `ProductionWorkflowApplication.from_backend`; a fully configured executor
factory is still an explicit deployment integration, not an implicit CLI default.
