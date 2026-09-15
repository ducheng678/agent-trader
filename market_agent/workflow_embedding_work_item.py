"""Closed host embedding task registry for graph and Harness observations."""

from __future__ import annotations

import re

from market_agent.workflow_observation import AttemptUsage, CoreNodeName


_SEMANTIC_TASK_ID = re.compile(r"semantic-embedding-[0-9a-f]{40}\Z")
_EMBEDDING_SOURCES = frozenset({"embedding_response", "embedding_usage_unavailable"})


def embedding_task_allowed(task_id: str, node: CoreNodeName) -> bool:
    if node is CoreNodeName.PLAN:
        return task_id in {"historical-query", "core-memory-embedding"}
    if node in {CoreNodeName.DISPATCH, CoreNodeName.RECOVER}:
        return bool(_SEMANTIC_TASK_ID.fullmatch(task_id))
    return False


def verified_embedding_work_item(
    task_id: str, node: CoreNodeName, attempts: tuple[AttemptUsage, ...],
) -> bool:
    return (
        embedding_task_allowed(task_id, node)
        and len(attempts) == 1
        and attempts[0].task_id == task_id
        and attempts[0].node is node
        and attempts[0].attempt == 0
        and attempts[0].source in _EMBEDDING_SOURCES
    )
