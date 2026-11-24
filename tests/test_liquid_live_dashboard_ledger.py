"""The live tables read events written by the real Ledger (flat rows, numpy booleans stored as "True"/"False")."""
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np

from research.liquid.live import dashboard as D
from research.liquid.live.ledger import Ledger

ET = ZoneInfo("America/New_York")
DAY = "2026-11-24"
REPORT = {"entries": {"live": [], "research": []}, "research_trades": [], "costs": [],
          "safety": {"ok": True, "kills": [], "max_unprotected_s": 0.0}}


def ledger_day(tmp_path) -> Ledger:
    """A flat day as the runner logs it: a plan, then two checks with numpy-bool `enter` and no entry."""
    led = Ledger(tmp_path / "c3_ledger.sqlite")
    led.log(datetime(2026, 11, 24, 8, 0, 13, tzinfo=ET), DAY, "plan", weight=0.86, vol=0.0232, prev_close=722.42)
    led.log(datetime(2026, 11, 24, 10, 0, 5, tzinfo=ET), DAY, "check", d=29, price=719.85, ub=726.02,
            vwap=714.46, enter=np.bool_(False), stop=726.02, in_position=False, n_carried=0)
    led.log(datetime(2026, 11, 24, 15, 0, 5, tzinfo=ET), DAY, "check", d=329, price=731.0, ub=730.84,
            vwap=732.0, enter=np.bool_(False), stop=730.84, in_position=False, n_carried=0)
    return led


def test_live_tables_read_real_ledger_events(tmp_path) -> None:
    led = ledger_day(tmp_path)
    assert led.events(DAY)[1]["enter"] == "False"          # how the runner's numpy bool lands in the ledger
    t = D.live_tables([DAY], {DAY: led.trades(DAY)}, {DAY: D.nested_events(led.events(DAY))},
                      {DAY: REPORT}, {DAY: {}})
    assert [(c["time"], c["action"]) for c in t["checks"]] == [("10:00", "inside"), ("15:00", "below_vwap")]
    (day,) = t["days"]
    assert day["trades"] == 0 and day["pnl"] == 0.0 and day["weight"] == 0.86 and day["safety_ok"]


def test_nested_events_keeps_every_field_and_parses_stored_booleans(tmp_path) -> None:
    rows = ledger_day(tmp_path).events(DAY)
    out = D.nested_events(rows)
    assert [e["kind"] for e in out] == ["plan", "check", "check"]
    assert all(set(e) == {"ts", "day", "kind", "payload"} for e in out)
    chk = out[1]["payload"]
    assert chk["enter"] is False and chk["in_position"] is False and chk["d"] == 29 and chk["price"] == 719.85
    assert D.nested_events([{"ts": "t", "day": DAY, "kind": "check", "enter": "True"}])[0]["payload"]["enter"] is True
