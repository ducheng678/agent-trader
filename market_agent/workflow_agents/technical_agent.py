from market_agent.workflow_contracts import TaskType
from market_agent.workflow_agents import common

PROMPT_PROFILE_ID = "workflow.technical.v1"


def _checked(task):
    if task.task_type is not TaskType.TECHNICAL:
        raise ValueError("task does not belong to this specialist")
    return task


def build_messages(task, context):
    return common.build_messages(_checked(task), context)


def build_invocation(task, context, **options):
    return common.build_invocation(_checked(task), context, **options)


def run_node(task, context, driver, **options):
    return common.run_node(_checked(task), context, driver, **options)
