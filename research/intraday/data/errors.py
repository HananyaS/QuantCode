"""Errors raised by the intraday data slice."""
from __future__ import annotations


class SealedHoldoutError(ValueError):
    """A session date inside the sealed holdout was requested without ``unseal=True``."""


class LookAheadError(AssertionError):
    """An input row is at or after the decision time it is supposed to precede."""


class MissingDataError(ValueError):
    """A required cache (e.g. the daily history before a session) is absent."""
