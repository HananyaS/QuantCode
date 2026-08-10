"""probe_data_feasibility.py — Phase 0 HARD GATE for the penny-stock veto research.

Purpose
-------
Answer, with real network calls and in under an hour, whether the whole research
programme is possible on free data.  The single question that matters most:

    Does the free Alpaca tier return SIP (consolidated) volume, or only IEX?

IEX is roughly 2.5% of consolidated US equity volume.  Relative volume (RVOL)
versus trailing ADV is the most load-bearing feature in the entire design, so
IEX-only volume does not degrade the research — it kills it.  If probe
``alpaca_sip_volume`` fails, the RVOL feature family is dead and the project
must be re-scoped before any further code is written.

The other probes establish whether the supporting free sources are reachable and
parseable.  They are informative, not gating.

Design notes
------------
* Every probe is wrapped so that one failure cannot abort the run.
* Nothing is written outside ``data/research_penny/`` and the verdict file.
* No repo module is imported, so this cannot perturb the live pipeline.
* Exit code is 0 if the hard gate passes, 1 if it fails — usable in CI later.

Usage
-----
    python research/penny/scripts/probe_data_feasibility.py
    python research/penny/scripts/probe_data_feasibility.py --verdict-dir outputs/penny
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import zipfile
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from ftplib import FTP
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import requests
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("probe")

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

# SEC requires a descriptive User-Agent with contact info on every request.
_SEC_UA = "QuantCode penny research shacharhananya@gmail.com"
_SEC_RATE_LIMIT_SLEEP = 0.15  # ~7 req/s, comfortably under the 10 req/s ceiling

# IEX is ~2.5% of consolidated volume.  If the SIP feed is silently falling back
# to IEX, the two feeds return identical bars.  We require SIP volume to exceed
# IEX volume by this factor on liquid names to consider SIP genuinely enabled.
_MIN_SIP_TO_IEX_VOLUME_RATIO = 3.0

# Liquid, long-listed names used to test feed behaviour.  Feed comparison is
# ticker-agnostic, so these need only be reliably present.
_FEED_TEST_TICKERS = ["AAPL", "F", "INTC"]

_NASDAQ_TRADED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt"
_NASDAQ_HALTS_URL = "https://www.nasdaqtrader.com/rss.aspx?feed=tradehalts"
_FINRA_SHORTVOL_URL = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{yyyymmdd}.txt"
_EDGAR_FTS_URL = "https://efts.sec.gov/LATEST/search-index"
_EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
_SEC_SUSPENSIONS_URL = "https://www.sec.gov/litigation/suspensions.htm"
_IBKR_FTP_HOST = "ftp3.interactivebrokers.com"
_IBKR_FTP_USER = "shortstock"
_IBKR_FTP_FILE = "usa.txt"
_STOOQ_BULK_URL = "https://stooq.com/db/d/?b=d_us_txt"

_HTTP_TIMEOUT = 60

# Project root, derived from this file's location: research/penny/scripts/<file>
_PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Credential resolution order. A dedicated research key is preferred; the live
# strategy keys are an accepted fallback because this module only ever builds a
# read-only StockHistoricalDataClient and never a TradingClient.
_CREDENTIAL_PAIRS: List[Tuple[str, str]] = [
    ("ALPACA_API_KEY_RESEARCH", "ALPACA_SECRET_KEY_RESEARCH"),
    ("ALPACA_API_KEY", "ALPACA_SECRET_KEY"),
    ("ALPACA_API_KEY_KELLY", "ALPACA_SECRET_KEY_KELLY"),
    ("ALPACA_API_KEY_LINEAR", "ALPACA_SECRET_KEY_LINEAR"),
]


def resolve_alpaca_credentials() -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Return the first credential pair that authenticates, plus its source name.

    Credentials in ``.env`` can be stale (a revoked key returns HTTP 401 on every
    feed, which is easily mistaken for a missing SIP entitlement -- a 403). This
    walks the candidates and returns the first that actually works.
    """
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    start, end = _recent_window(lookback_days=7)
    for key_name, secret_name in _CREDENTIAL_PAIRS:
        key, secret = os.environ.get(key_name), os.environ.get(secret_name)
        if not key or not secret:
            continue
        try:
            client = StockHistoricalDataClient(api_key=key, secret_key=secret)
            df = client.get_stock_bars(
                StockBarsRequest(
                    symbol_or_symbols=["AAPL"],
                    timeframe=TimeFrame.Day,
                    start=start,
                    end=end,
                    feed="iex",
                )
            ).df
            if not df.empty:
                logger.info("credentials: using %s", key_name)
                return key, secret, key_name
            logger.warning("credentials: %s authenticated but returned no data", key_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("credentials: %s rejected (%s)", key_name, str(exc)[:80])
    return None, None, None


# --------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------


@dataclass
class ProbeResult:
    """Outcome of a single feasibility probe."""

    name: str
    passed: bool
    gating: bool
    detail: str
    data: Dict[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> str:
        if self.passed:
            return "PASS"
        return "FAIL (GATE)" if self.gating else "fail"


def _safe(name: str, gating: bool, fn: Callable[[], Tuple[bool, str, Dict[str, Any]]]) -> ProbeResult:
    """Run a probe, converting any exception into a failed result."""
    logger.info("probe: %s ...", name)
    try:
        passed, detail, data = fn()
    except Exception as exc:  # noqa: BLE001 - a probe must never abort the run
        logger.warning("probe %s raised: %s", name, exc)
        return ProbeResult(name, False, gating, f"raised {type(exc).__name__}: {exc}")
    result = ProbeResult(name, passed, gating, detail, data)
    logger.info("probe: %s -> %s  (%s)", name, result.status, detail)
    return result


def _recent_window(lookback_days: int = 30, lag_days: int = 3) -> Tuple[str, str]:
    """A historical date window ending safely in the past.

    ``lag_days`` keeps ``end`` well beyond the 15-minute delay that gates free
    SIP access, so a 403 means "no SIP entitlement" rather than "too recent".
    """
    end = date.today() - timedelta(days=lag_days)
    start = end - timedelta(days=lookback_days)
    return start.isoformat(), end.isoformat()


# --------------------------------------------------------------------------
# Probe 1 (GATE): Alpaca SIP vs IEX volume
# --------------------------------------------------------------------------


def _fetch_alpaca_daily(
    tickers: List[str], start: str, end: str, feed: str, api_key: str, secret_key: str
) -> pd.DataFrame:
    """Fetch daily bars for one feed. Returns a (symbol, timestamp) MultiIndex frame."""
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)
    request = StockBarsRequest(
        symbol_or_symbols=tickers,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        adjustment="all",
        feed=feed,
    )
    return client.get_stock_bars(request).df


def probe_alpaca_sip_volume(api_key: str, secret_key: str) -> Tuple[bool, str, Dict[str, Any]]:
    """THE GATE. Compare SIP vs IEX reported volume on liquid names.

    If ``feed="sip"`` is honoured, consolidated volume should be many multiples
    of IEX volume.  If the two are equal, the account is silently IEX-only and
    RVOL cannot be computed from Alpaca.
    """
    start, end = _recent_window()

    try:
        sip = _fetch_alpaca_daily(_FEED_TEST_TICKERS, start, end, "sip", api_key, secret_key)
    except Exception as exc:  # noqa: BLE001
        return False, f"SIP request rejected ({type(exc).__name__}: {exc})", {"sip_error": str(exc)}

    iex = _fetch_alpaca_daily(_FEED_TEST_TICKERS, start, end, "iex", api_key, secret_key)

    if sip.empty:
        return False, "SIP returned an empty frame", {}
    if iex.empty:
        return False, "IEX returned an empty frame (cannot form a baseline)", {}

    ratios: Dict[str, float] = {}
    for symbol in _FEED_TEST_TICKERS:
        if symbol not in sip.index.get_level_values("symbol"):
            continue
        if symbol not in iex.index.get_level_values("symbol"):
            continue
        sip_v = sip.xs(symbol, level="symbol")["volume"]
        iex_v = iex.xs(symbol, level="symbol")["volume"]
        common = sip_v.index.intersection(iex_v.index)
        if len(common) == 0:
            continue
        iex_total = float(iex_v.loc[common].sum())
        if iex_total <= 0:
            continue
        ratios[symbol] = float(sip_v.loc[common].sum()) / iex_total

    if not ratios:
        return False, "no overlapping SIP/IEX bars to compare", {}

    median_ratio = float(pd.Series(ratios).median())
    passed = median_ratio >= _MIN_SIP_TO_IEX_VOLUME_RATIO
    detail = (
        f"median SIP/IEX volume ratio = {median_ratio:.1f}x "
        f"(threshold {_MIN_SIP_TO_IEX_VOLUME_RATIO}x) over {start}..{end}"
    )
    if not passed:
        detail += " -- SIP appears to be falling back to IEX; RVOL is NOT computable from Alpaca"
    return passed, detail, {"ratios": ratios, "median_ratio": median_ratio, "window": [start, end]}


def probe_alpaca_vs_yfinance(api_key: str, secret_key: str) -> Tuple[bool, str, Dict[str, Any]]:
    """Cross-check Alpaca SIP volume against yfinance consolidated volume.

    Confirms SIP volume is not merely *larger* than IEX but actually correct.
    """
    import yfinance as yf

    start, end = _recent_window()
    sip = _fetch_alpaca_daily(_FEED_TEST_TICKERS, start, end, "sip", api_key, secret_key)
    if sip.empty:
        return False, "no SIP data to compare", {}

    errors: Dict[str, float] = {}
    for symbol in _FEED_TEST_TICKERS:
        if symbol not in sip.index.get_level_values("symbol"):
            continue
        alpaca_v = sip.xs(symbol, level="symbol")["volume"]
        alpaca_v.index = pd.to_datetime(alpaca_v.index.tz_convert("America/New_York").date)

        yf_df = yf.download(
            symbol, start=start, end=end, progress=False, auto_adjust=True
        )
        if yf_df is None or yf_df.empty:
            continue
        yf_v = yf_df["Volume"]
        if isinstance(yf_v, pd.DataFrame):
            yf_v = yf_v.iloc[:, 0]
        yf_v.index = pd.to_datetime(yf_v.index.date)

        common = alpaca_v.index.intersection(yf_v.index)
        if len(common) < 5:
            continue
        yf_total = float(yf_v.loc[common].sum())
        if yf_total <= 0:
            continue
        errors[symbol] = abs(float(alpaca_v.loc[common].sum()) - yf_total) / yf_total

    if not errors:
        return False, "no overlapping bars for cross-check", {}

    worst = max(errors.values())
    passed = worst < 0.10
    return (
        passed,
        f"max |Alpaca SIP - yfinance| / yfinance volume = {worst:.1%} across {len(errors)} names",
        {"relative_errors": errors},
    )


# --------------------------------------------------------------------------
# Probe 2: Nasdaq Trader symbol directory + halts
# --------------------------------------------------------------------------


def probe_nasdaq_symbol_directory() -> Tuple[bool, str, Dict[str, Any]]:
    """The point-in-time listed universe, snapshotted forward each day."""
    resp = requests.get(_NASDAQ_TRADED_URL, timeout=_HTTP_TIMEOUT)
    resp.raise_for_status()
    text = resp.text

    # Pipe-delimited; final line is a "File Creation Time" trailer.
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines or "|" not in lines[0]:
        return False, "unexpected format (no pipe delimiter in header)", {}
    trailer = lines[-1] if lines[-1].lower().startswith("file creation") else None
    body = lines[1:-1] if trailer else lines[1:]

    df = pd.read_csv(io.StringIO("\n".join([lines[0]] + body)), sep="|")
    cols = set(df.columns)
    required = {"Symbol", "ETF", "Test Issue"}
    if not required.issubset(cols):
        return False, f"missing expected columns; got {sorted(cols)}", {"columns": sorted(cols)}

    tradable = df[(df["Test Issue"] == "N") & (df["ETF"] == "N")]
    return (
        True,
        f"{len(df)} symbols, {len(tradable)} non-ETF non-test; trailer={trailer!r}",
        {"total": int(len(df)), "tradable": int(len(tradable)), "columns": sorted(cols)},
    )


def probe_nasdaq_halts() -> Tuple[bool, str, Dict[str, Any]]:
    """LULD halt feed. No free history exists, so this must be recorded forward."""
    resp = requests.get(_NASDAQ_HALTS_URL, timeout=_HTTP_TIMEOUT)
    resp.raise_for_status()
    body = resp.text
    if "<rss" not in body[:2000].lower() and "<?xml" not in body[:200].lower():
        return False, "response is not XML/RSS", {"head": body[:200]}
    n_items = body.lower().count("<item>")
    return (
        True,
        f"RSS reachable, {n_items} halt items in the current feed "
        f"(empty outside market hours is normal)",
        {"n_items": n_items},
    )


# --------------------------------------------------------------------------
# Probe 3: FINRA daily short-sale volume
# --------------------------------------------------------------------------


def probe_finra_short_volume() -> Tuple[bool, str, Dict[str, Any]]:
    """Consolidated NMS short volume. Free, with history back to 2018-08-01."""
    attempts: List[str] = []
    for lag in range(2, 12):  # walk back to skip weekends/holidays
        day = date.today() - timedelta(days=lag)
        if day.weekday() >= 5:
            continue
        url = _FINRA_SHORTVOL_URL.format(yyyymmdd=day.strftime("%Y%m%d"))
        attempts.append(day.isoformat())
        resp = requests.get(url, timeout=_HTTP_TIMEOUT)
        if resp.status_code != 200 or not resp.text.strip():
            continue
        df = pd.read_csv(io.StringIO(resp.text), sep="|")
        if "Symbol" not in df.columns:
            continue
        return (
            True,
            f"{day.isoformat()}: {len(df)} rows, columns={list(df.columns)}",
            {"date": day.isoformat(), "rows": int(len(df)), "columns": list(df.columns)},
        )
    return False, f"no file found for any of {attempts}", {"attempted": attempts}


# --------------------------------------------------------------------------
# Probe 4: SEC EDGAR
# --------------------------------------------------------------------------


def probe_edgar_fulltext() -> Tuple[bool, str, Dict[str, Any]]:
    """Full-text search drives the dilution and deficiency-notice veto features."""
    headers = {"User-Agent": _SEC_UA, "Accept-Encoding": "gzip, deflate"}
    params = {"q": '"substantial doubt about its ability to continue"', "forms": "10-Q"}
    resp = requests.get(_EDGAR_FTS_URL, params=params, headers=headers, timeout=_HTTP_TIMEOUT)
    resp.raise_for_status()
    payload = resp.json()
    total = payload.get("hits", {}).get("total", {}).get("value")
    if total is None:
        return False, f"unexpected payload shape: {list(payload)[:5]}", {"keys": list(payload)}
    return (
        True,
        f"going-concern query returned {total} 10-Q hits",
        {"total_hits": int(total)},
    )


def probe_edgar_submissions() -> Tuple[bool, str, Dict[str, Any]]:
    """Per-company filing index: 8-K items, S-3/424B5, Form 25 delistings."""
    headers = {"User-Agent": _SEC_UA, "Accept-Encoding": "gzip, deflate"}
    apple_cik = 320193
    resp = requests.get(
        _EDGAR_SUBMISSIONS_URL.format(cik=apple_cik), headers=headers, timeout=_HTTP_TIMEOUT
    )
    resp.raise_for_status()
    payload = resp.json()
    recent = payload.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    if not forms:
        return False, "no filings array in submissions payload", {}
    return (
        True,
        f"{payload.get('name')}: {len(forms)} recent filings, fields={sorted(recent)[:6]}",
        {"n_recent": len(forms), "fields": sorted(recent)},
    )


def probe_sec_suspensions() -> Tuple[bool, str, Dict[str, Any]]:
    """Trading-suspension history feeds the prior-manipulation veto feature."""
    headers = {"User-Agent": _SEC_UA}
    resp = requests.get(_SEC_SUSPENSIONS_URL, headers=headers, timeout=_HTTP_TIMEOUT)
    resp.raise_for_status()
    try:
        tables = pd.read_html(io.StringIO(resp.text))
    except ValueError:
        return False, "page reachable but contains no parseable table", {}
    if not tables:
        return False, "no tables parsed", {}
    biggest = max(tables, key=len)
    return (
        True,
        f"{len(tables)} table(s), largest has {len(biggest)} rows, cols={list(biggest.columns)[:4]}",
        {"n_tables": len(tables), "rows": int(len(biggest))},
    )


# --------------------------------------------------------------------------
# Probe 5: IBKR borrow file (recorder-only; no free history)
# --------------------------------------------------------------------------


def probe_ibkr_borrow() -> Tuple[bool, str, Dict[str, Any]]:
    """Shortable-shares + borrow fee. Snapshot only -- must be recorded daily."""
    buf: List[bytes] = []
    try:
        ftp = FTP(_IBKR_FTP_HOST, timeout=45)
    except OSError as exc:
        return (
            False,
            f"cannot reach {_IBKR_FTP_HOST} ({exc}) -- FTP is commonly blocked by "
            f"corporate/ISP firewalls; needs an alternate route or a different network",
            {"error": str(exc)},
        )
    try:
        ftp.login(user=_IBKR_FTP_USER, passwd=_IBKR_FTP_USER)
        ftp.set_pasv(True)  # active-mode data channels are usually firewalled
        ftp.retrbinary(f"RETR {_IBKR_FTP_FILE}", buf.append, blocksize=32768)
    finally:
        try:
            ftp.quit()
        except Exception:  # noqa: BLE001
            ftp.close()

    raw = b"".join(buf).decode("latin-1", errors="replace")
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    if len(lines) < 10:
        return False, f"file too short ({len(lines)} lines)", {"head": lines[:3]}
    # Format: "#BOF" header, then pipe-delimited rows, "#EOF" trailer.
    data_lines = [ln for ln in lines if "|" in ln and not ln.startswith("#")]
    return (
        True,
        f"{len(data_lines)} borrow rows; header={lines[0][:80]!r}",
        {"rows": len(data_lines), "header": lines[0][:200], "sample": data_lines[1:3]},
    )


# --------------------------------------------------------------------------
# Probe 6: Stooq bulk (survivorship backstop)
# --------------------------------------------------------------------------


def probe_stooq_bulk() -> Tuple[bool, str, Dict[str, Any]]:
    """Stooq bulk US daily includes delisted tickers -- the survivorship backstop.

    Stooq throttles aggressively and may serve an HTML interstitial instead of
    the archive.  A non-zip response is a soft failure, not a blocker: the
    archive can be fetched manually once and cached.
    """
    resp = requests.get(
        _STOOQ_BULK_URL,
        timeout=_HTTP_TIMEOUT,
        headers={"User-Agent": "Mozilla/5.0 (research)"},
        stream=True,
    )
    ctype = resp.headers.get("Content-Type", "")
    clen = resp.headers.get("Content-Length")
    head = resp.raw.read(4, decode_content=True)
    resp.close()

    if head[:2] == b"PK":
        return True, f"zip archive served (Content-Length={clen})", {"content_type": ctype}
    return (
        False,
        f"non-zip response (Content-Type={ctype!r}); likely a throttle page -- "
        f"fetch manually once and cache",
        {"content_type": ctype, "head": head.decode("latin-1", errors="replace")},
    )


# --------------------------------------------------------------------------
# Verdict rendering
# --------------------------------------------------------------------------


def render_verdict(results: List[ProbeResult], cred_source: Optional[str] = None) -> str:
    """Render a human-readable verdict document."""
    gate = [r for r in results if r.gating]
    gate_ok = all(r.passed for r in gate)
    n_pass = sum(r.passed for r in results)

    lines: List[str] = [
        "# Phase 0 — Data Feasibility Verdict",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        f"**HARD GATE: {'PASS' if gate_ok else 'FAIL'}** "
        f"({n_pass}/{len(results)} probes passed overall)",
        "",
        f"Alpaca credentials: `{cred_source or 'none'}`",
        "",
    ]

    if gate_ok:
        lines += [
            "Alpaca's free tier serves consolidated (SIP) volume for historical",
            "queries. Relative volume is computable, so the RVOL feature family is",
            "viable and the research programme can proceed to Phase 1.",
            "",
        ]
    else:
        lines += [
            "> **STOP.** The gating probe failed. Consolidated volume is not",
            "> available from the free tier, so RVOL cannot be computed as designed.",
            "> Do not build Phase 1+ until the universe/feature scope is revised.",
            "",
        ]

    lines += ["| Probe | Gating | Status | Detail |", "|---|---|---|---|"]
    for r in results:
        detail = r.detail.replace("|", "\\|")
        lines.append(f"| `{r.name}` | {'yes' if r.gating else 'no'} | {r.status} | {detail} |")

    failed_soft = [r for r in results if not r.passed and not r.gating]
    if failed_soft:
        lines += ["", "## Non-gating failures", ""]
        for r in failed_soft:
            lines.append(f"- **{r.name}** — {r.detail}")

    lines += [
        "",
        "## Notes carried forward",
        "",
        "- Borrow (IBKR) and halts (Nasdaq) have **no free history**. Whatever is",
        "  not recorded from today onward is permanently unavailable at zero cost.",
        "- FINRA consolidated short volume begins 2018-08-01; that date bounds any",
        "  retrospective study using short-volume features.",
        "- SEC endpoints require a descriptive User-Agent and a <10 req/s cap.",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 0 data feasibility gate.")
    parser.add_argument(
        "--verdict-dir",
        default="outputs/penny",
        help="Directory for the verdict file (default: outputs/penny).",
    )
    args = parser.parse_args(argv)

    # find_dotenv() walks up from the *calling frame*, which does not reliably
    # reach the project root. Load the known path explicitly.
    load_dotenv(_PROJECT_ROOT / ".env")
    api_key, secret_key, cred_source = resolve_alpaca_credentials()

    results: List[ProbeResult] = []

    if not api_key or not secret_key:
        tried = ", ".join(k for k, _ in _CREDENTIAL_PAIRS)
        results.append(
            ProbeResult(
                "alpaca_sip_volume",
                False,
                True,
                f"no working Alpaca credentials found (tried: {tried})",
            )
        )
    else:
        logger.info("using credentials from %s", cred_source)
        results.append(
            _safe("alpaca_sip_volume", True, lambda: probe_alpaca_sip_volume(api_key, secret_key))
        )
        results.append(
            _safe(
                "alpaca_vs_yfinance",
                False,
                lambda: probe_alpaca_vs_yfinance(api_key, secret_key),
            )
        )

    results.append(_safe("nasdaq_symbol_directory", False, probe_nasdaq_symbol_directory))
    results.append(_safe("nasdaq_halts", False, probe_nasdaq_halts))
    results.append(_safe("finra_short_volume", False, probe_finra_short_volume))
    results.append(_safe("edgar_fulltext", False, probe_edgar_fulltext))
    results.append(_safe("edgar_submissions", False, probe_edgar_submissions))
    results.append(_safe("sec_suspensions", False, probe_sec_suspensions))
    results.append(_safe("ibkr_borrow", False, probe_ibkr_borrow))
    results.append(_safe("stooq_bulk", False, probe_stooq_bulk))

    verdict = render_verdict(results, cred_source)

    verdict_dir = Path(args.verdict_dir)
    verdict_dir.mkdir(parents=True, exist_ok=True)
    (verdict_dir / "phase0_verdict.md").write_text(verdict, encoding="utf-8")
    (verdict_dir / "phase0_probes.json").write_text(
        json.dumps([r.__dict__ for r in results], indent=2, default=str), encoding="utf-8"
    )

    print()
    print(verdict)
    print(f"\nWritten to {verdict_dir / 'phase0_verdict.md'}")

    gate_ok = all(r.passed for r in results if r.gating)
    return 0 if gate_ok else 1


if __name__ == "__main__":
    sys.exit(main())
