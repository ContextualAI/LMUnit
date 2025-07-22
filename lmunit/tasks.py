from typing import Callable, Optional

from .metrics import (calculate_accuracy, calculate_correlation_human_score,
                      calculate_lfqa, calculate_rewardbench_v1)

# Task registry to store all registered tasks
_TASK_REGISTRY = {}


def eval_task(name: Optional[str] = None):
    """Decorator to register evaluation task classes.

    Args:
        name: Optional name for the task. If not provided, uses the class name.

    """

    def decorator(cls):
        task_name = name or cls.__name__
        _TASK_REGISTRY[task_name] = cls
        cls._task_name = task_name
        return cls

    return decorator


def get_task(name: str):
    """Get a registered task by name."""
    if name not in _TASK_REGISTRY:
        raise ValueError(
            f"Task '{name}' not found. Available tasks: {list(_TASK_REGISTRY.keys())}"
        )
    return _TASK_REGISTRY[name]


def list_tasks():
    """List all registered task names."""
    return list(_TASK_REGISTRY.keys())


@eval_task("infobench")
class InfoBenchExpertSplit:
    dataset_name: str = "ContextualAI/InfoBenchExpertSplit"
    split: str = "test"
    mode: str = "direct"
    metric_fn: Callable = calculate_accuracy
    use_rubric: bool = False
    use_reference: bool = False
    threshold: float = 2.5


@eval_task("flask")
class Flask:
    dataset_name: str = "ContextualAI/Flask"
    split: str = "test"
    mode: str = "direct"
    metric_fn: Callable = calculate_correlation_human_score
    use_rubric: bool = True
    use_reference: bool = True
    threshold: float = 2.5


@eval_task("biggenbench")
class BigGenBench:
    dataset_name: str = "ContextualAI/BigGenBench"
    split: str = "test"
    mode: str = "direct"
    metric_fn: Callable = calculate_correlation_human_score
    use_rubric: bool = True
    use_reference: bool = True
    threshold: float = 2.5


@eval_task("lfqa")
class LFQA:
    dataset_name: str = "ContextualAI/LFQA"
    split: str = "test"
    mode: str = "preference"
    column_1 = "response_a"
    column_2 = "response_b"
    use_rubric: bool = False
    use_reference: bool = False
    metric_fn: Callable = calculate_lfqa
    threshold: float = None


@eval_task("rewardbenchv1")
class RewardBenchV1:
    dataset_name: str = "ContextualAI/reward-bench-filtered"
    split: str = "test"
    mode: str = "preference"
    column_1 = "chosen_response"
    column_2 = "rejected_response"
    use_rubric: bool = False
    use_reference: bool = False
    metric_fn: Callable = calculate_rewardbench_v1
    threshold: float = None
