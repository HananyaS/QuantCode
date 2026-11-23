"""The C3 broker's paper-endpoint guard against the real alpaca-py client (no network: the clients only
store their base URL at construction)."""
import pytest

from research.liquid.live import broker as B


def test_paper_client_passes_the_guard() -> None:
    b = B.AlpacaPaperBroker("PKTESTKEY", "testsecret")
    assert B.paper_endpoint(b._c._base_url).startswith(B.PAPER_HOST)


def test_live_endpoint_is_refused(monkeypatch) -> None:
    from alpaca.trading import client as tc
    real = tc.TradingClient

    def live(key: str, secret: str, paper: bool = True, **kw):
        return real(key, secret, paper=False, **kw)
    monkeypatch.setattr(tc, "TradingClient", live)
    with pytest.raises(AssertionError, match="non-paper endpoint"):
        B.AlpacaPaperBroker("AKTESTKEY", "testsecret")


@pytest.mark.parametrize("base, want", [("https://paper-api.alpaca.markets", "https://paper-api.alpaca.markets"),
                                        ("", "")])
def test_paper_endpoint_reads_plain_strings(base: str, want: str) -> None:
    assert B.paper_endpoint(base) == want


def test_shorting_config_reads_and_patches_the_raw_endpoint(monkeypatch) -> None:
    b = B.AlpacaPaperBroker("PKTESTKEY", "testsecret")
    state = {"closing_transactions_only": False, "fractional_trading": True, "max_margin_multiplier": "4",
             "no_shorting": False, "suspend_trade": False}       # the paper API's 2026-11-24 reply: no dtbp/pdt_check
    sent = []
    monkeypatch.setattr(b._c, "get", lambda path, data=None: dict(state) if path == B.CONFIG_PATH else None)

    def patch(path, data=None):
        sent.append((path, data))
        state.update(data)
        return dict(state)
    monkeypatch.setattr(b._c, "patch", patch)
    assert b.shorting_enabled() is True
    b.disable_shorting()
    assert sent == [(B.CONFIG_PATH, {"no_shorting": True})]
    assert b.shorting_enabled() is False


def test_config_without_the_shorting_flag_is_refused(monkeypatch) -> None:
    b = B.AlpacaPaperBroker("PKTESTKEY", "testsecret")
    monkeypatch.setattr(b._c, "get", lambda path, data=None: {"shorting_enabled": False})
    monkeypatch.setattr(B, "READ_RETRIES", 1, raising=False)
    with pytest.raises(AssertionError, match="unexpected account configuration"):
        b.shorting_enabled()
