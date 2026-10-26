"""Sealed-holdout guard: no session date >= HOLDOUT_START is read without a recorded decision."""
from __future__ import annotations

from datetime import date

from research.intraday.data.errors import SealedHoldoutError

HOLDOUT_START: date = date(2025, 8, 7)


def check_holdout(session_date: date, holdout_start: date = HOLDOUT_START,
                  unseal: bool = False) -> None:
    """Raise ``SealedHoldoutError`` for ``session_date >= holdout_start`` unless ``unseal``."""
    if session_date >= holdout_start and not unseal:
        raise SealedHoldoutError(
            f"session date {session_date.isoformat()} is inside the sealed holdout "
            f"(holdout_start={holdout_start.isoformat()}); "
            "pass unseal=True only by a recorded decision")
