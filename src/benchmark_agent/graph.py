from typing import Annotated, Optional, TypedDict, Dict, Any

from langgraph.graph import StateGraph, START, END
from langgraph.pregel.types import default_retry_on
from langgraph.types import RetryPolicy

from benchmark_agent.nodes import (
    generate_pandas_node, approve_pandas_node, run_pandas_docker_node,
    generate_dask_node, approve_dask_node, run_dask_docker_node,
    generate_numpy_node, approve_numpy_node, run_numpy_docker_node,
    generate_polars_node, approve_polars_node, run_polars_docker_node,
    merge_results_node, generate_report_node,
    make_approval_router, make_execution_router
)


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------
# One nested BranchState per library under its own top-level key. The 4-way
# fan-in into merge_results is conflict-free because each branch only ever
# writes its own key; the merge_branch reducer exists for the *sequential*
# partial updates within a branch (without it, LangGraph's default reducer
# would replace the whole sub-dict on every node update, e.g. approve_<lib>
# writing only "decision" would wipe "code").

class BranchState(TypedDict, total=False):
    code: str
    reasoning: str
    model: str               # model that generated the code ("provider:model")
    decision: str            # "approve" | "regenerate" | "reject"
    attempts: int            # total generations (display only)
    exec_attempts: int       # runtime-error fixes (capped at MAX_EXECUTION_FIXES)
    exec_error: Optional[str]  # last runtime error; None after a successful run
    metrics: Dict[str, Any]


def merge_branch(left: Optional[BranchState], right: Optional[BranchState]) -> BranchState:
    """Reducer: merges partial branch updates instead of overwriting the dict."""
    return {**(left or {}), **(right or {})}


Branch = Annotated[BranchState, merge_branch]


class GraphState(TypedDict, total=False):
    analytical_request: str

    # Per-library branch state
    pandas: Branch
    dask: Branch
    numpy: Branch
    polars: Branch

    # Merged & Output State
    results: Dict[str, Any]
    selected_strategy: str


# ---------------------------------------------------------------------------
# Retry policies
# ---------------------------------------------------------------------------

def _llm_retry_on(exc: Exception) -> bool:
    # Provider request timeouts (aiohttp/asyncio TimeoutError) subclass
    # OSError, which default_retry_on blanket-refuses, but a timed-out LLM
    # call is exactly the transient failure this policy exists for. Everything
    # else keeps the default behavior (ValueError stays fail-fast for the
    # missing-request guard in the generate nodes).
    return isinstance(exc, TimeoutError) or default_retry_on(exc)


llm_retry = RetryPolicy(
    max_attempts=3,
    initial_interval=2.0,
    backoff_factor=2.0,
    max_interval=30.0,
    jitter=True,
    retry_on=_llm_retry_on,
)

# No RetryPolicy on the docker nodes: run_in_docker never raises (every
# failure is returned as a metrics dict), so a policy there would never fire.


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------

workflow = StateGraph(GraphState)

# Pandas Branch Nodes
workflow.add_node("generate_pandas", generate_pandas_node, retry_policy=llm_retry)
workflow.add_node("approve_pandas", approve_pandas_node)
workflow.add_node("run_pandas_docker", run_pandas_docker_node)

# Dask Branch Nodes
workflow.add_node("generate_dask", generate_dask_node, retry_policy=llm_retry)
workflow.add_node("approve_dask", approve_dask_node)
workflow.add_node("run_dask_docker", run_dask_docker_node)

# NumPy Branch Nodes
workflow.add_node("generate_numpy", generate_numpy_node, retry_policy=llm_retry)
workflow.add_node("approve_numpy", approve_numpy_node)
workflow.add_node("run_numpy_docker", run_numpy_docker_node)

# Polars Branch Nodes
workflow.add_node("generate_polars", generate_polars_node, retry_policy=llm_retry)
workflow.add_node("approve_polars", approve_polars_node)
workflow.add_node("run_polars_docker", run_polars_docker_node)

# Merge & Report Nodes
# defer=True: wait for ALL parallel branches to complete before executing,
# even when branches have different lengths (e.g. rejection/regeneration loops).
workflow.add_node("merge_results", merge_results_node, defer=True)
workflow.add_node("generate_report", generate_report_node)

# Fan-Out: START -> 4 parallel branches
workflow.add_edge(START, "generate_pandas")
workflow.add_edge(START, "generate_dask")
workflow.add_edge(START, "generate_numpy")
workflow.add_edge(START, "generate_polars")

# Per-branch chains. After approve_<lib> a conditional router decides:
# "regenerate" -> back to generate_<lib> (no limit); "approve"/"reject" ->
# run_<lib>_docker (reject skips execution and records the failure). After
# run_<lib>_docker a second router sends runtime failures of approved code
# back to generate_<lib> with the error (up to MAX_EXECUTION_FIXES times);
# otherwise the branch proceeds to merge_results.
for _lib in ("pandas", "dask", "numpy", "polars"):
    workflow.add_edge(f"generate_{_lib}", f"approve_{_lib}")
    workflow.add_conditional_edges(
        f"approve_{_lib}",
        make_approval_router(_lib),
        [f"generate_{_lib}", f"run_{_lib}_docker"],
    )
    workflow.add_conditional_edges(
        f"run_{_lib}_docker",
        make_execution_router(_lib),
        [f"generate_{_lib}", "merge_results"],
    )

# Fan-In from all branches and output report
workflow.add_edge("merge_results", "generate_report")
workflow.add_edge("generate_report", END)

# Compile graph
graph = workflow.compile()
