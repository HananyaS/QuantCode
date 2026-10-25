"""Architectural tripwires for research/intraday (AST walks, no imports of the code)."""
from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
INTRADAY = REPO_ROOT / "research" / "intraday"
TRADING_ALLOWLIST = {"research/intraday/exec/alpaca_paper.py"}
AGENT_PACKAGES = ("data", "state", "model", "risk", "exec", "ops")
_CLOCK_METHODS = {"now", "utcnow", "today"}
_CLOCK_OWNERS = {"datetime", "date"}
_TIME_FUNCS = {"time", "monotonic"}


def _rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _py_files() -> list[Path]:
    return sorted(INTRADAY.rglob("*.py"))


def _trading_imports(source: str) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source)):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.append(node.module)
            names += [a.name for a in node.names]
        for name in names:
            if name.startswith("alpaca.trading") or name.endswith("TradingClient"):
                found.append((node.lineno, name))
    return found


def _wall_clock_calls(source: str) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        attr, owner = node.func.attr, node.func.value
        owner_name = owner.id if isinstance(owner, ast.Name) else (
            owner.attr if isinstance(owner, ast.Attribute) else None)
        if attr in _CLOCK_METHODS and owner_name in _CLOCK_OWNERS:
            found.append((node.lineno, f"{owner_name}.{attr}()"))
        elif attr in _TIME_FUNCS and isinstance(owner, ast.Name) and owner.id == "time":
            found.append((node.lineno, f"time.{attr}()"))
    return found


def _sibling_imports(source: str, pkg: str, rel_parts: tuple[str, ...]) -> list[tuple[int, str]]:
    """Imports of another agent package. rel_parts = module path under research/intraday."""
    prefix = "research.intraday"
    found: list[tuple[int, str]] = []

    def check(dotted: str, lineno: int) -> None:
        parts = dotted.split(".")
        if dotted.startswith(prefix + ".") and len(parts) > 2:
            other = parts[2]
            if other in AGENT_PACKAGES and other != pkg:
                found.append((lineno, dotted))

    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for a in node.names:
                check(a.name, node.lineno)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = list(rel_parts[:-1])  # package of this module under intraday
                base = base[: len(base) - (node.level - 1)] if node.level > 1 else base
                mod = ".".join([prefix, *base] + ([node.module] if node.module else []))
            else:
                mod = node.module or ""
            check(mod, node.lineno)
            if mod == prefix:
                for a in node.names:
                    check(f"{prefix}.{a.name}", node.lineno)
    return found


def test_intraday_package_exists() -> None:
    assert (INTRADAY / "contracts" / "__init__.py").is_file()


def test_no_trading_client_import_outside_allowlist() -> None:
    offenders = [
        f"{_rel(p)}:{ln}: {what}"
        for p in _py_files() if _rel(p) not in TRADING_ALLOWLIST
        for ln, what in _trading_imports(p.read_text(encoding="utf-8"))
    ]
    assert offenders == []


def test_no_wall_clock_calls() -> None:
    offenders = [
        f"{_rel(p)}:{ln}: {what}"
        for p in _py_files()
        for ln, what in _wall_clock_calls(p.read_text(encoding="utf-8"))
    ]
    assert offenders == []


def test_agent_packages_do_not_import_each_other() -> None:
    from research.intraday import AGENT_PACKAGES as declared

    assert tuple(declared) == AGENT_PACKAGES
    offenders: list[str] = []
    for p in _py_files():
        rel_parts = p.relative_to(INTRADAY).with_suffix("").parts
        if len(rel_parts) < 2 or rel_parts[0] not in AGENT_PACKAGES:
            continue
        for ln, what in _sibling_imports(p.read_text(encoding="utf-8"), rel_parts[0], rel_parts):
            offenders.append(f"{_rel(p)}:{ln}: {what}")
    assert offenders == []


# ---- self-tests: a tripwire that cannot fire is itself a defect -------------
def test_selftest_trading_detector_fires() -> None:
    assert _trading_imports("from alpaca.trading.client import TradingClient\n")
    assert _trading_imports("import alpaca.trading\n")
    assert _trading_imports("from foo import TradingClient\n")
    assert not _trading_imports("from alpaca.data import StockHistoricalDataClient\n")


def test_selftest_wall_clock_detector_fires() -> None:
    for src in ("datetime.now()", "datetime.datetime.now()", "date.today()",
                "datetime.utcnow()", "time.time()", "time.monotonic()"):
        assert _wall_clock_calls(f"x = {src}\n"), src
    assert not _wall_clock_calls("x = bar.ts.now_offset()\ny = time.sleep(1)\n")


def test_selftest_isolation_detector_fires() -> None:
    parts = ("model", "m")
    assert _sibling_imports("from research.intraday.data import x\n", "model", parts)
    assert _sibling_imports("import research.intraday.risk.y\n", "model", parts)
    assert _sibling_imports("from research.intraday import risk\n", "model", parts)
    assert _sibling_imports("from ..data import x\n", "model", parts)
    assert not _sibling_imports("from research.intraday.contracts import x\n", "model", parts)
    assert not _sibling_imports("from research.intraday.model import x\n", "model", parts)
