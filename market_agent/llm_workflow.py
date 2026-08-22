from __future__ import annotations

from typing import Any, Callable, Dict, TypedDict

from langgraph.graph import END, START, StateGraph


class WorkflowState(TypedDict, total=False):
    call: Callable[[], Any]
    judge: Callable[[], Any]
    should_price: Callable[[Any], bool]
    price: Callable[[Any], Any]
    assemble: Callable[[Any, Any], Any]
    judge_result: Any
    pricing_result: Any
    result: Any


def _run_single(state: WorkflowState) -> Dict[str, Any]:
    return {"result": state["call"]()}


def _judge(state: WorkflowState) -> Dict[str, Any]:
    return {"judge_result": state["judge"]()}


def _route_after_judge(state: WorkflowState) -> str:
    return "price" if state["should_price"](state["judge_result"]) else "assemble"


def _price(state: WorkflowState) -> Dict[str, Any]:
    return {"pricing_result": state["price"](state["judge_result"])}


def _assemble(state: WorkflowState) -> Dict[str, Any]:
    return {
        "result": state["assemble"](
            state["judge_result"],
            state.get("pricing_result"),
        )
    }


def _build_single_graph():
    builder = StateGraph(WorkflowState)
    builder.add_node("model_call", _run_single)
    builder.add_edge(START, "model_call")
    builder.add_edge("model_call", END)
    return builder.compile()


def _build_passive_graph():
    builder = StateGraph(WorkflowState)
    builder.add_node("judge", _judge)
    builder.add_node("price", _price)
    builder.add_node("assemble", _assemble)
    builder.add_edge(START, "judge")
    builder.add_conditional_edges("judge", _route_after_judge)
    builder.add_edge("price", "assemble")
    builder.add_edge("assemble", END)
    return builder.compile()


class LLMWorkflow:
    def __init__(self):
        self._single = _build_single_graph()
        self._passive = _build_passive_graph()

    def run_single(self, call: Callable[[], Any]) -> Any:
        return self._single.invoke({"call": call})["result"]

    def run_passive(
        self,
        *,
        judge: Callable[[], Any],
        should_price: Callable[[Any], bool],
        price: Callable[[Any], Any],
        assemble: Callable[[Any, Any], Any],
    ) -> Any:
        return self._passive.invoke(
            {
                "judge": judge,
                "should_price": should_price,
                "price": price,
                "assemble": assemble,
            }
        )["result"]
