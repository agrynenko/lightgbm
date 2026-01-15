import os
import re
from typing import List, Tuple


def _load_log_text(log_or_path: str) -> str:
    """
    If input is a file path, read it.
    Otherwise assume it's raw log text.
    """
    if os.path.isfile(log_or_path):
        with open(log_or_path, "r", encoding="utf-8") as f:
            return f.read()
    return log_or_path


def _extract_metric_and_time(
    log_text: str,
    time_pattern: str,
    metric_pattern: str
) -> Tuple[List[float], List[float]]:
    """
    Generic helper to extract iteration times and metric values.

    Parameters
    ----------
    log_text : str
        Full training log text.
    time_pattern : str
        Regex with ONE capture group for time in seconds.
    metric_pattern : str
        Regex with ONE capture group for metric value.

    Returns
    -------
    times : List[float]
        Per-iteration elapsed times.
    metrics : List[float]
        Per-iteration metric values.
    """
    time_regex = re.compile(time_pattern)
    metric_regex = re.compile(metric_pattern)

    times = [float(m.group(1)) for m in time_regex.finditer(log_text)]
    metrics = [float(m.group(1)) for m in metric_regex.finditer(log_text)]

    return times, metrics


def parse_lightgbm_log(
    log_or_path: str,
    metric_name: str
) -> Tuple[List[float], List[float]]:
    """
    Extract iteration times and metric values from LightGBM logs.
    Accepts either a file path or raw log text.
    """
    log_text = _load_log_text(log_or_path)

    time_pattern = r"([0-9]+\.[0-9]+)\s+seconds elapsed"
    metric_pattern = rf"{re.escape(metric_name)}\s*:\s*([0-9]+\.[0-9]+)"

    return _extract_metric_and_time(log_text, time_pattern, metric_pattern)


def parse_xgboost_log(
    log_or_path: str,
    metric_name: str
) -> Tuple[List[float], List[float]]:
    """
    Extract iteration times and metric values from XGBoost logs.
    Accepts either a file path or raw log text.
    """
    log_text = _load_log_text(log_or_path)

    time_pattern = r"took\s+([0-9]+\.[0-9]+)\s+seconds"
    metric_pattern = rf"{re.escape(metric_name)}:([0-9]+\.[0-9]+)"

    return _extract_metric_and_time(log_text, time_pattern, metric_pattern)

