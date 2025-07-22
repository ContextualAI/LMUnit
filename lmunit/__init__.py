"""LMUnit - Language Model Unit Testing Framework."""

from .lmunit import LMUnit
from .metrics import (calculate_accuracy, calculate_correlation_human_score,
                      calculate_lfqa, calculate_rewardbench_v1)
from .tasks import eval_task, get_task, list_tasks

__version__ = "0.1.0"
__all__ = [
    "LMUnit",
    "eval_task",
    "get_task",
    "list_tasks",
    "calculate_accuracy",
    "calculate_correlation_human_score",
    "calculate_lfqa",
    "calculate_rewardbench_v1",
]
