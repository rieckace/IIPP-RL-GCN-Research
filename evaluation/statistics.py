"""Statistics helpers for evaluation reporting."""

from __future__ import annotations

from math import sqrt
from typing import Dict, Tuple


def success_rate_confidence_interval(
	successes: int,
	total: int,
	confidence_z: float = 1.96,
) -> Tuple[float, float]:
	"""Return a normal-approximation confidence interval for a success rate."""
	if total <= 0:
		return 0.0, 0.0

	rate = successes / total
	standard_error = sqrt(max(rate * (1.0 - rate), 0.0) / total)
	margin = confidence_z * standard_error
	return max(0.0, rate - margin), min(1.0, rate + margin)


def summarize_outcomes(outcomes: Dict[str, int], total: int) -> Dict[str, float]:
	"""Convert raw outcome counts into percentages."""
	if total <= 0:
		return {}

	return {reason: count / total for reason, count in outcomes.items()}
