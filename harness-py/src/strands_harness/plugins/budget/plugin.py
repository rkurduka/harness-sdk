"""Budget plugin: one-line budget enforcement for Strands agents."""

import asyncio
import logging
from typing import TYPE_CHECKING

from strands.hooks import AfterInvocationEvent, BeforeModelCallEvent
from strands.interventions import Deny
from strands.plugins import Plugin
from strands.storage import LocalFileStorage, Storage
from pathlib import Path
           

from strands_harness.plugins.budget.aws_pricing import fetch_aws_rates
from strands_harness.plugins.budget.budget_state import BudgetManager
from strands_harness.plugins.budget.intervention import BudgetIntervention

if TYPE_CHECKING:
    from strands.agent.agent import Agent

logger = logging.getLogger(__name__)


class BudgetPlugin(Plugin):
    """Enforces a persistent spending budget on an agent.

    Attach to an agent to cap model-call spending. Costs are estimated from
    token usage and recorded to storage, so budgets persist across restarts.
    When the budget is exhausted, further model calls are blocked.

    Budgets are keyed by ``session_id``. If not provided, the agent's
    ``agent_id`` is used (default: ``"default"``), so persistence works even
    without a session manager.

    A pricing source is required for enforcement: pass ``custom_rates``
    (keyed by exact model ID) and/or ``use_aws_pricing=True``. Without one,
    a warning is logged and model calls are not billed. Changing or
    resetting the budget is deliberately not exposed to the model — use the
    Python API: ``await plugin.manager.set_budget(...)``.

    Example:
        ```python
        from strands import Agent
        from strands_budget import BudgetPlugin

        plugin = BudgetPlugin(
            budget=10.0,
            session_id="my-project",
            custom_rates={"qwen.qwen3-coder-next": (0.00000022, 0.0000018)},
        )
        agent = Agent(model="qwen.qwen3-coder-next", plugins=[plugin])
        ```
    """

    name = "budget-plugin"

    def __init__(
        self,
        budget: float | None = None,
       # session_id: str | None = None,
        storage: Storage | None = None,
        custom_rates: dict[str, tuple[float, float]] | None = None,
        use_aws_pricing: bool = False,
        aws_region: str = "us-east-1",
    ) -> None:
        """Initialize the plugin.

        Args:
            budget: Budget limit in USD. Creates or updates the persisted limit
                on attach. None leaves any existing budget unchanged (and means
                unlimited if no budget was ever set).
            session_id: Stable identifier keying the budget in storage. Defaults
                to the agent's ``agent_id`` at attach time.
            storage: Storage backend for budget state. Defaults to local files
                under ``.agent``.
            custom_rates: Per-token ``{model_substring: (input_rate, output_rate)}``
                overrides. Highest precedence: they win over AWS-fetched rates
                and the built-in defaults.
            use_aws_pricing: Fetch current Bedrock token prices from the AWS
                Price List API on attach (requires ``pricing:GetProducts``
                permission). Covers many model families but not recent Anthropic
                Claude models; uncovered models keep their static/custom rates.
                Fetch failures fall back to the static table with a warning.
            aws_region: The Bedrock region whose prices to fetch.
        """

        super().__init__()
        self._pricing_configured = bool(custom_rates or use_aws_pricing)
        if not self._pricing_configured:
            logger.warning(
                "BudgetPlugin has no pricing source (custom_rates or use_aws_pricing). "
                "Model calls will NOT be billed, so the budget limit will never be "
                "enforced. Pass custom_rates={...} or use_aws_pricing=True to enable "
                "budget enforcement."
            )
        if use_aws_pricing and custom_rates:
            logger.warning(
                "Pass either custom_rates or use_aws_pricing, In case both provided , "
                "plugin will look for aws pricing for model."
            )

        self.manager = BudgetManager(storage=storage)
        self._budget = budget
        #self._configured_session_id = session_id
        self._custom_rates = custom_rates
        self._use_aws_pricing = use_aws_pricing
        self._aws_region = aws_region
        #self._session_id: str | None = None
        self._intervention: BudgetIntervention | None = None

    @property
    def session_id(self) -> str | None:
        """The resolved session ID (available after attach to an agent)."""
        return self._session_id

    async def init_agent(self, agent: "Agent") -> None:
        """Attach budget enforcement to the agent.

        Resolves the session ID, applies the configured budget limit, and
        registers the model-call hooks that enforce and meter spending.
        """

        session_id = agent.session_id
        print("SESSION ID %s", session_id)
        if session_id:
            base_dir = Path(".agent")
            session_path = base_dir / "sessions" / "session" / session_id
            logger.info("Session '%s' uses storage dir: %s", session_id, session_path)
            agent_session_storage = LocalFileStorage(session_path)
            self.manager = BudgetManager(storage=agent_session_storage)

        rates = self._custom_rates
        if self._use_aws_pricing:
            # boto3 is synchronous; keep the event loop free while fetching.
            aws_rates = await asyncio.to_thread(fetch_aws_rates, self._aws_region)
            # custom_rates keep the last word over fetched prices
            rates = {**aws_rates, **(self._custom_rates or {})}

        self._intervention = BudgetIntervention(self.manager, session_id, rates)

        if self._budget is not None:
            state = await self.manager.load(session_id)
            if state is None or state.total_budget != self._budget:
                await self.manager.set_budget(session_id, self._budget)
                logger.info("Budget for session '%s' set to $%.2f", session_id, self._budget)

        agent.add_hook(self._before_model, BeforeModelCallEvent)
        agent.add_hook(self._after_invocation, AfterInvocationEvent)

    async def _before_model(self, event: BeforeModelCallEvent) -> None:
        """Cancel the model call if the budget is exhausted. Fails closed."""
        assert self._intervention is not None  # set in init_agent
        try:
            decision = await self._intervention.before_model_call(event)
        except Exception:
            logger.exception("Budget check failed; blocking model call (fail closed)")
            event.cancel = "Budget check failed; model call blocked."
            return
        if isinstance(decision, Deny):
            event.cancel = decision.reason

    async def _after_invocation(self, event: AfterInvocationEvent) -> None:
        """Bill the final model cycle when the invocation ends.

        Mid-invocation cycles are billed by the before-model hook; usage from
        the last cycle only appears in the metrics after the loop finishes.
        """
        assert self._intervention is not None  # set in init_agent
        try:
            await self._intervention.record_usage(event.agent)
        except Exception:
            logger.exception("Failed to record model call cost")
