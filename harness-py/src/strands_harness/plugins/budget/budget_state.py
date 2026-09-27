"""Budget state models and manager for persisting spend across sessions."""

import os
from datetime import datetime, timezone

from pydantic import BaseModel, Field
from strands.storage import LocalFileStorage, Storage

DEFAULT_STORAGE_DIR = os.path.expanduser(".agent")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Transaction(BaseModel):
    """A single recorded cost event."""

    description: str
    cost: float
    timestamp: str = Field(default_factory=_utc_now_iso)
    input_tokens: int = 0
    output_tokens: int = 0


class BudgetState(BaseModel):
    """The persisted budget for a single session."""

    session_id: str
    total_budget: float
    spent: float = 0.0
    transactions: list[Transaction] = Field(default_factory=list)

    @property
    def remaining(self) -> float:
        """Amount left to spend."""
        return self.total_budget - self.spent

    def is_exhausted(self) -> bool:
        """True when spending has reached or exceeded the budget."""
        return self.spent >= self.total_budget


class BudgetManager:
    """Loads and saves budget state through any Strands ``Storage`` backend.

    A missing budget means unlimited spending: ``load`` returns ``None`` and
    ``record_cost``/``reset`` are no-ops for sessions without a budget.
    """

    def __init__(self, storage: Storage | None = None) -> None:
        self.storage = storage or LocalFileStorage(DEFAULT_STORAGE_DIR)

    def _key(self, session_id: str) -> str:
        return f"strands-budget/{session_id}.json"

    async def load(self, session_id: str) -> BudgetState | None:
        """Return the budget for a session, or None if no budget is set.

        Corrupt data raises ``pydantic.ValidationError`` rather than being
        silently discarded.
        """
        data = await self.storage.read(self._key(session_id))
        if data is None:
            return None
        return BudgetState.model_validate_json(data)

    async def save(self, state: BudgetState) -> None:
        """Persist the budget state as JSON bytes."""
        await self.storage.write(self._key(state.session_id), state.model_dump_json(indent=2).encode())

    async def set_budget(self, session_id: str, amount: float) -> BudgetState:
        """Create a budget, or update the limit while preserving spend history."""
        state = await self.load(session_id)
        if state is None:
            state = BudgetState(session_id=session_id, total_budget=amount)
        else:
            state.total_budget = amount
        await self.save(state)
        return state

    async def record_cost(
        self,
        session_id: str,
        cost: float,
        description: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> BudgetState | None:
        """Add a transaction and increase spent. No-op if no budget is set."""
        state = await self.load(session_id)
        if state is None:
            return None
        state.spent += round(cost, 6)  # avoid floating point errors
        state.transactions.append(
            Transaction(
                description=description,
                cost=cost,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        )
        await self.save(state)
        return state

    async def reset(self, session_id: str) -> BudgetState | None:
        """Zero out spending and history, keeping the budget limit. No-op if no budget is set."""
        state = await self.load(session_id)
        if state is None:
            return None
        state.spent = 0.00000
        state.transactions = []
        await self.save(state)
        return state

    async def delete(self, session_id: str) -> None:
        """Remove the budget entirely (spending becomes unlimited)."""
        await self.storage.delete(self._key(session_id))
