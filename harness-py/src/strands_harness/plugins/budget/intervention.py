"""Budget intervention: blocks model calls when the session budget is exhausted."""

import logging
from typing import Any

from strands.hooks.events import BeforeModelCallEvent
from strands.interventions import Deny, InterventionHandler, OnError, Proceed

from strands_harness.plugins.budget.budget_state import BudgetManager
from strands_harness.plugins.budget.pricing import estimate_cost

logger = logging.getLogger(__name__)


class BudgetIntervention(InterventionHandler):
    """Enforces a per-session spending budget on model calls.

    Before each model call, any newly consumed tokens (tracked as a delta
    against the agent's accumulated usage) are priced and recorded, then the
    session budget is checked and the call is denied if spending has reached
    the limit.

    Billing happens in ``before_model_call`` rather than ``after_model_call``
    because the SDK updates ``accumulated_usage`` only after the after-model
    event has fired. The final cycle of an invocation must therefore be billed
    by calling ``record_usage(agent)`` when the invocation ends — the
    ``BudgetPlugin`` does this via an ``AfterInvocationEvent`` hook. If you use
    this handler standalone (via ``Agent(interventions=[...])``), leftover
    usage is billed at the start of the next invocation instead.

    Sessions without a budget are unlimited: all operations no-op.

    Note: output tokens can't be known before a call, so the call that
    crosses the limit completes; the *next* call is blocked.

    Example:
        ```python
        from strands import Agent
        from strands_budget import BudgetIntervention, BudgetManager

        manager = BudgetManager()
        await manager.set_budget("my-session", 10.0)
        agent = Agent(interventions=[BudgetIntervention(manager, "my-session")])
        ```
    """

    name = "budget-intervention"

    def __init__(
        self,
        manager: BudgetManager,
        session_id: str,
        rates: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        """Initialize the intervention.

        Args:
            manager: Budget manager used to load state and record costs.
            session_id: Stable identifier keying the budget in storage.
            rates: Per-token ``{model_id: (input_rate, output_rate)}`` table,
                looked up by exact model ID. Models without an entry are not
                billed.
        """
        self.manager = manager
        self.session_id = session_id
        self.rates = rates or {}
        # Accumulated token counts already billed. after_model_call bills
        # only the delta between the agent's running totals and these.
        self._billed_input_tokens = 0
        self._billed_output_tokens = 0

    @property
    def on_error(self) -> OnError:
        """Fail closed: if budget checks error out, block the call."""
        return "deny"

    async def before_model_call(self, event: BeforeModelCallEvent, **kwargs: Any) -> Proceed | Deny:
        """Bill any unbilled usage, then deny the call if the budget is exhausted."""
        # The SDK updates accumulated_usage only after AfterModelCallEvent has
        # fired, so the previous cycle's tokens are billed here, just before
        # the next call — keeping mid-invocation enforcement accurate.
        await self.record_usage(event.agent)
        state = await self.manager.load(self.session_id)
        if state is None:
            return Proceed()  # no budget set = unlimited
        if state.is_exhausted():
            reason = (
                f"Budget exhausted for session '{self.session_id}': "
                f"spent ${state.spent:.5f} of ${state.total_budget:.5f}"
            )
            logger.critical(reason)
            return Deny(reason=reason)
        return Proceed()

    async def record_usage(self, agent: Any) -> None:
        """Bill any tokens accumulated since the last billing.

        Reads the agent's accumulated usage metrics, computes the delta since
        the previous call, prices it, and records a transaction. Called before
        each model call and again when the invocation ends (the SDK only
        updates usage metrics after ``AfterModelCallEvent`` has fired).
        """
        usage = agent.event_loop_metrics.accumulated_usage
        input_total = usage.get("inputTokens", 0)
        output_total = usage.get("outputTokens", 0)

        new_input = input_total - self._billed_input_tokens
        new_output = output_total - self._billed_output_tokens

        # Advance the anchors before awaiting storage so a failed
        # record_cost can't lead to double billing on the next call.
        self._billed_input_tokens = input_total
        self._billed_output_tokens = output_total

        if new_input <= 0 and new_output <= 0:
            return  # nothing new to bill

        model_id = agent.model.get_config().get("model_id", "unknown")
        if not model_id:
            logger.warning("No model ID found in agent config, skipping billing")
            return  # no model ID = can't bill

        model_rates = self.rates.get(model_id, None)
        if not model_rates:
            logger.warning(f"No rate found for model {model_id}, skipping billing")
            return  # no rate = can't bill

        cost = estimate_cost(new_input, new_output, model_rates)
        state = await self.manager.record_cost(
            self.session_id,
            cost,
            f"model call ({model_id})",
            input_tokens=new_input,
            output_tokens=new_output,
        )
        if state is not None:
            logger.debug(
                "Billed $%.6f (%d in / %d out tokens); remaining $%.6f",
                cost,
                new_input,
                new_output,
                state.remaining,
            )
