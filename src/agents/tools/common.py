"""
Helpers shared by the agent tools: error handling and compact, LLM-friendly output.

Tools never raise on bad input. They return {"status": "error", "error": "..."}
so the agent can correct the call or explain the problem to the user.
"""
import functools
import logging
import math
from typing import Any, Callable, Dict

from services.feature_input import FlowInputError

logger = logging.getLogger(__name__)

EXPECTED_ERRORS = (FlowInputError, KeyError, ValueError, FileNotFoundError)


def error_result(message: str) -> Dict[str, Any]:
    return {"status": "error", "error": message}


def safe_tool(func: Callable[..., Dict[str, Any]]) -> Callable[..., Dict[str, Any]]:
    """Turn expected input errors into an error result instead of an exception."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except EXPECTED_ERRORS as exc:
            message = exc.args[0] if isinstance(exc, KeyError) and exc.args else str(exc)
            return error_result(str(message))
        except Exception as exc:  # unexpected: log with traceback, still return a usable result
            logger.exception(f"Tool {func.__name__} failed")
            return error_result(f"Internal error in {func.__name__}: {exc}")

    return wrapper


def rounded(value: Any, digits: int = 4) -> Any:
    """Recursively round floats so tool output stays short and readable."""
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return round(value, digits)
    if isinstance(value, dict):
        return {k: rounded(v, digits) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [rounded(v, digits) for v in value]
    return value


def compact_input(input_summary: Dict[str, Any]) -> Dict[str, Any]:
    """Keep what the agent needs to judge input quality; drop the long 'provided' list."""
    return {
        "input_quality": input_summary["input_quality"],
        "known_metrics": f"{input_summary['known_count']}/24",
        "derived": input_summary["derived"],
        "filled_with_normal_values": input_summary["filled"],
        "warnings": input_summary["warnings"],
    }


def compact_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """ML service result in a shorter form for the LLM."""
    out = dict(result)
    if out.get("status") == "needs_follow_up":
        # Nothing was predicted, so the filled list is noise; keep what is missing and the question.
        out["input"] = {
            "input_quality": out["input"]["input_quality"],
            "known_metrics": f"{out['input']['known_count']}/24",
            "provided": out["input"]["provided"],
        }
    elif "input" in out:
        out["input"] = compact_input(out["input"])
    return rounded(out)
