"""Architectural tripwires for research/intraday (AST walks, no imports of the code)."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
INTRADAY = REPO_ROOT / "research" / "intraday"
TRADING_ALLOWLIST = {"research/intraday/exec/alpaca_paper.py"}
AGENT_PACKAGES = ("data", "state", "model", "risk", "exec", "ops")
_CLOCK_METHODS = {"now", "utcnow", "today"}
_CLOCK_OWNERS = {"datetime", "date", "Timestamp"}
_TIME_FUNCS = {f"time.{f}" for f in (
    "time", "time_ns", "perf_counter", "perf_counter_ns", "monotonic", "monotonic_ns",
    "localtime", "gmtime")}
_NOW_STRING_CTORS = {"Timestamp", "to_datetime", "datetime64"}
_NOW_STRINGS = {"now", "today"}
_ENTROPY_FUNCS = {"uuid.uuid1", "uuid.uuid4", "os.urandom"}
# Conventional aliases, applied when a snippet uses them without an import.
_CANONICAL = {"np": "numpy", "pd": "pandas"}
_STDLIB = frozenset(sys.stdlib_module_names)
# Layering: what each core layer may import besides the standard library
# (a layer may always import itself).
_LAYER_ALLOWED: dict[str, tuple[str, ...]] = {
    "contracts": ("research.intraday.contracts",),
    "engine": ("research.intraday.contracts", "research.intraday.engine",
               "research.intraday.ops.recorder"),
}


def _rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _py_files() -> list[Path]:
    return sorted(INTRADAY.rglob("*.py"))


def _aliases(tree: ast.AST) -> dict[str, str]:
    """Local name -> fully qualified dotted name, from the module's absolute imports."""
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.asname:
                    out[a.asname] = a.name
                else:
                    top = a.name.split(".")[0]
                    out[top] = top
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            for a in node.names:
                if a.name != "*":
                    out[a.asname or a.name] = f"{node.module}.{a.name}"
    return out


def _dotted(expr: ast.AST, aliases: dict[str, str]) -> str | None:
    if isinstance(expr, ast.Name):
        return aliases.get(expr.id, _CANONICAL.get(expr.id, expr.id))
    if isinstance(expr, ast.Attribute):
        base = _dotted(expr.value, aliases)
        return None if base is None else f"{base}.{expr.attr}"
    return None


def _calls(source: str) -> list[tuple[ast.Call, str]]:
    """Every call whose callee resolves to a dotted name, aliases expanded."""
    tree = ast.parse(source)
    aliases = _aliases(tree)
    out: list[tuple[ast.Call, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _dotted(node.func, aliases)
            if name is not None:
                out.append((node, name))
    return out


def _first_str_arg(call: ast.Call) -> str | None:
    if call.args and isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, str):
        return call.args[0].value
    return None


def _trading_imports(source: str) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []

    def check(name: str, lineno: int) -> None:
        if name.startswith("alpaca.trading") or name.endswith("TradingClient"):
            found.append((lineno, name))

    tree = ast.parse(source)
    aliases = _aliases(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                check(a.name, node.lineno)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                check(node.module, node.lineno)
            for a in node.names:
                check(f"{node.module}.{a.name}" if node.module else a.name, node.lineno)
        elif isinstance(node, ast.Attribute):
            dotted = _dotted(node, aliases)
            if dotted and dotted.startswith("alpaca.trading"):
                found.append((node.lineno, dotted))
    for call, name in _calls(source):
        target = _first_str_arg(call)
        if (name.endswith("import_module") or name == "__import__") and target                 and target.startswith("alpaca.trading"):
            found.append((call.lineno, f"{name}({target!r})"))
    return found


def _wall_clock_calls(source: str) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    for call, name in _calls(source):
        parts = name.split(".")
        if len(parts) >= 2 and parts[-1] in _CLOCK_METHODS and parts[-2] in _CLOCK_OWNERS:
            found.append((call.lineno, f"{name}()"))
        elif name in _TIME_FUNCS:
            found.append((call.lineno, f"{name}()"))
        elif parts[-1] in _NOW_STRING_CTORS:
            arg = _first_str_arg(call)
            if arg is not None and arg.strip().lower() in _NOW_STRINGS:
                found.append((call.lineno, f"{name}({arg!r})"))
    return found


def _unseeded(call: ast.Call) -> bool:
    if not call.args and not call.keywords:
        return True
    seed = call.args[0] if call.args else next(
        (k.value for k in call.keywords if k.arg == "seed"), None)
    return isinstance(seed, ast.Constant) and seed.value is None


def _nondeterminism_calls(source: str) -> list[tuple[int, str]]:
    """Entropy sources: uuid1/uuid4, os.urandom, secrets, global random, unseeded rng."""
    found: list[tuple[int, str]] = []
    for call, name in _calls(source):
        parts = name.split(".")
        if name in _ENTROPY_FUNCS or (parts[0] == "secrets" and len(parts) > 1):
            found.append((call.lineno, f"{name}()"))
        elif parts[0] == "random" and len(parts) == 2 and parts[1] != "Random":
            found.append((call.lineno, f"{name}() (module-level random state)"))
        elif parts[-2:] == ["random", "default_rng"] and _unseeded(call):
            found.append((call.lineno, f"{name}() without a seed"))
    return found


def _resolve_from(node: ast.ImportFrom, rel_parts: tuple[str, ...]) -> str:
    """Absolute module of an ImportFrom; rel_parts = module path under research/intraday."""
    if not node.level:
        return node.module or ""
    base = list(rel_parts[:-1])
    base = base[: len(base) - (node.level - 1)] if node.level > 1 else base
    return ".".join(["research.intraday", *base] + ([node.module] if node.module else []))


def _layer_violations(source: str, rel_parts: tuple[str, ...]) -> list[tuple[int, str]]:
    """contracts/ imports only contracts + stdlib; engine/ adds only ops.recorder."""
    allowed = _LAYER_ALLOWED[rel_parts[0]]
    found: list[tuple[int, str]] = []

    def check(dotted: str, lineno: int) -> None:
        if dotted.split(".")[0] in _STDLIB:
            return
        if not any(dotted == a or dotted.startswith(a + ".") for a in allowed):
            found.append((lineno, dotted))

    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for a in node.names:
                check(a.name, node.lineno)
        elif isinstance(node, ast.ImportFrom):
            mod = _resolve_from(node, rel_parts)
            for a in node.names:
                check(f"{mod}.{a.name}", node.lineno)
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


# ---- strengthened tripwires (final-review fix A6) ---------------------------
import pytest  # noqa: E402

_WALL_CLOCK_SOURCES = [
    "import pandas as pd\nx = pd.Timestamp.now()\n",
    "import pandas as pd\nx = pd.Timestamp.utcnow()\n",
    "import pandas as pd\nx = pd.Timestamp.today()\n",
    "import pandas as pd\nx = pd.Timestamp('now')\n",
    "import pandas as pd\nx = pd.Timestamp('today')\n",
    "from pandas import Timestamp\nx = Timestamp('now')\n",
    "import pandas as pd\nx = pd.to_datetime('now')\n",
    "import numpy as np\nx = np.datetime64('now')\n",
    "import numpy as np\nx = np.datetime64('today')\n",
    "from datetime import datetime as dt\nx = dt.now()\n",
    "import datetime as d\nx = d.datetime.utcnow()\n",
    "from datetime import date as D\nx = D.today()\n",
    "from time import time\nx = time()\n",
    "import time as t\nx = t.time()\n",
    "import time\nx = time.time_ns()\n",
    "import time\nx = time.perf_counter()\n",
    "import time\nx = time.perf_counter_ns()\n",
    "import time\nx = time.monotonic_ns()\n",
    "from time import monotonic as m\nx = m()\n",
    "import time\nx = time.localtime()\n",
    "import time\nx = time.gmtime()\n",
]


@pytest.mark.parametrize("src", _WALL_CLOCK_SOURCES)
def test_selftest_wall_clock_patterns(src: str) -> None:
    assert _wall_clock_calls(src), src


def test_selftest_wall_clock_negative_controls() -> None:
    for src in ("import pandas as pd\nx = pd.Timestamp('2024-03-04T14:31Z')\n",
                "import numpy as np\nx = np.datetime64('2024-03-04')\n",
                "import time\ntime.sleep(0)\n",
                "from datetime import datetime\nx = datetime(2024, 3, 4)\n"):
        assert not _wall_clock_calls(src), src


_NONDETERMINISM_SOURCES = [
    "import uuid\nx = uuid.uuid1()\n",
    "import uuid\nx = uuid.uuid4()\n",
    "from uuid import uuid4\nx = uuid4()\n",
    "import os\nx = os.urandom(8)\n",
    "from os import urandom\nx = urandom(8)\n",
    "import secrets\nx = secrets.token_hex(8)\n",
    "from secrets import choice\nx = choice([1, 2])\n",
    "import random\nx = random.random()\n",
    "import random\nrandom.shuffle(xs)\n",
    "from random import choice\nx = choice([1, 2])\n",
    "import random as r\nx = r.randint(0, 9)\n",
    "import numpy as np\nrng = np.random.default_rng()\n",
    "import numpy as np\nrng = np.random.default_rng(None)\n",
    "from numpy.random import default_rng\nrng = default_rng()\n",
]


@pytest.mark.parametrize("src", _NONDETERMINISM_SOURCES)
def test_selftest_nondeterminism_patterns(src: str) -> None:
    assert _nondeterminism_calls(src), src


def test_selftest_nondeterminism_negative_controls() -> None:
    for src in ("import numpy as np\nrng = np.random.default_rng(0)\n",
                "import numpy as np\nrng = np.random.default_rng(seed=7)\n",
                "import random\nrng = random.Random(42)\n",
                "import uuid\nx = uuid.uuid5(uuid.NAMESPACE_DNS, 'a')\n"):
        assert not _nondeterminism_calls(src), src


def test_no_nondeterminism_calls() -> None:
    offenders = [
        f"{_rel(p)}:{ln}: {what}"
        for p in _py_files()
        for ln, what in _nondeterminism_calls(p.read_text(encoding="utf-8"))
    ]
    assert offenders == []


@pytest.mark.parametrize("src", [
    "from alpaca import trading\n",
    "from alpaca.trading import client\n",
    "import alpaca.trading.client as c\n",
    "import importlib\nm = importlib.import_module('alpaca.trading')\n",
    "from importlib import import_module\nm = import_module('alpaca.trading.client')\n",
    "m = __import__('alpaca.trading')\n",
    "import alpaca as a\nc = a.trading.TradingClient\n",
])
def test_selftest_trading_patterns(src: str) -> None:
    assert _trading_imports(src), src


def test_selftest_trading_negative_controls() -> None:
    for src in ("from alpaca import data\n",
                "import importlib\nm = importlib.import_module('alpaca.data')\n"):
        assert not _trading_imports(src), src


@pytest.mark.parametrize("layer,src", [
    ("contracts", "import numpy as np\n"),
    ("contracts", "import pandas\n"),
    ("contracts", "from research.intraday.engine.bus import EventBus\n"),
    ("contracts", "from research.intraday import ops\n"),
    ("contracts", "from ..engine import bus\n"),
    ("engine", "import numpy\n"),
    ("engine", "from research.intraday.ops import JsonlRecorder\n"),
    ("engine", "from research.intraday.ops.other import x\n"),
    ("engine", "from research.intraday.risk import sizing\n"),
    ("engine", "import research.intraday.data\n"),
])
def test_selftest_layer_detector_fires(layer: str, src: str) -> None:
    assert _layer_violations(src, (layer, "m")), (layer, src)


def test_selftest_layer_detector_negative_controls() -> None:
    assert not _layer_violations(
        "from __future__ import annotations\nimport json\nfrom datetime import datetime\n"
        "from research.intraday.contracts.events import Event\nfrom . import events\n",
        ("contracts", "codec"))
    assert not _layer_violations(
        "import heapq\nfrom research.intraday.contracts import Event\n"
        "from research.intraday.ops.recorder import JsonlRecorder, log_hash\n"
        "from research.intraday.engine.bus import EventBus\nfrom .bus import EventBus\n",
        ("engine", "replay"))


def test_contracts_and_engine_layering() -> None:
    offenders: list[str] = []
    for p in _py_files():
        rel_parts = p.relative_to(INTRADAY).with_suffix("").parts
        if len(rel_parts) < 2 or rel_parts[0] not in _LAYER_ALLOWED:
            continue
        for ln, what in _layer_violations(p.read_text(encoding="utf-8"), rel_parts):
            offenders.append(f"{_rel(p)}:{ln}: {what}")
    assert offenders == []


def test_jsonl_is_binary_to_git() -> None:
    import subprocess
    out = subprocess.run(["git", "check-attr", "text", "--", "x.jsonl"], cwd=REPO_ROOT,
                         check=True, capture_output=True, text=True).stdout
    assert out.strip() == "x.jsonl: text: unset"
