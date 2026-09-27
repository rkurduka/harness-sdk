"""Cost estimation from token counts and per-token rates.

Rate lookup (by exact model ID) happens in ``intervention.py``; this module
only does the arithmetic. Rates are USD per single token as an
``(input_rate, output_rate)`` pair.
"""

import logging

logger = logging.getLogger(__name__)


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    rates: tuple[float, float],
) -> float:
    """Estimate the cost in USD of a model call based on token counts."""
    # round off to 5 decimal points
    return round(input_tokens * rates[0] + output_tokens * rates[1], 5)

