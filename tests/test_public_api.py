"""API-surface guarantees that a 1.0 makes, and the drift that broke them.

A stable release freezes more than the names in ``__all__``. Every private
helper another module reaches into becomes effectively public — undocumented,
untested as an API, and free to change under people who came to depend on it.
This module pins the four things the pre-1.0 audit found.
"""

from __future__ import annotations

import ast
import pathlib
import re

import pytest

import qorch
from qorch import Circuit


PACKAGE = pathlib.Path(qorch.__file__).parent
SOURCES = [p for p in PACKAGE.rglob("*.py") if "__pycache__" not in str(p)]


# ── 1. one version, derived not duplicated ───────────────────────────────


def test_version_matches_the_installed_distribution() -> None:
    """It lived in two files and drifted; it is now read from the metadata."""
    from importlib.metadata import version

    assert qorch.__version__ == version("qorch")


def test_the_version_is_not_hardcoded_twice() -> None:
    """A literal alongside the derived value is the drift waiting to happen."""
    source = (PACKAGE / "__init__.py").read_text(encoding="utf-8")
    literals = re.findall(r'^__version__\s*=\s*["\']', source, re.M)
    assert not literals, "__version__ should be derived, not assigned a literal"


# ── 2. no module reaches into another module's privates ──────────────────


def _cross_module_private_imports() -> list[str]:
    offences = []
    for path in SOURCES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("qorch"):
                for alias in node.names:
                    if alias.name.startswith("_") and not alias.name.startswith("__"):
                        offences.append(
                            f"{path.relative_to(PACKAGE)} imports "
                            f"{alias.name} from {node.module}"
                        )
    return offences


def test_no_module_imports_a_private_name_from_another_module() -> None:
    """The audit found five. A private another module needs is not private.

    Each was promoted to a public name with the old spelling kept as an alias:
    ``analysis.circuit_depth``, ``tomography.rotate_to_basis``,
    ``adp.combine_circuits``, ``tools.parse_circuit``, ``gates.xx_matrix``.
    """
    offences = _cross_module_private_imports()
    assert not offences, "cross-module private imports:\n  " + "\n  ".join(offences)


@pytest.mark.parametrize("module,public,private", [
    ("qorch.analysis", "circuit_depth", "_compute_depth"),
    ("qorch.tomography", "rotate_to_basis", "_rotate_to_basis"),
    ("qorch.adp", "combine_circuits", "_combine_circuits"),
    ("qorch.tools", "parse_circuit", "_parse_circuit"),
])
def test_promoted_helpers_keep_their_old_private_alias(module, public, private) -> None:
    """Promotion must not break anyone who was already importing the private."""
    import importlib

    mod = importlib.import_module(module)
    assert hasattr(mod, public), f"{module}.{public} is missing"
    assert getattr(mod, private) is getattr(mod, public)


def test_the_ms_matrix_lives_in_the_gate_registry() -> None:
    """It was private to the simulator and imported from there by a backend.

    A simulator internal being load-bearing for a hardware adapter is the exact
    coupling the gate registry exists to remove.
    """
    from qorch.backends.simulator import _xx_matrix
    from qorch.gates import xx_matrix

    assert _xx_matrix is xx_matrix
    assert len(xx_matrix(0.25)) == 16


# ── 3. building a gate by name is public ─────────────────────────────────


def test_circuit_gate_builds_by_name() -> None:
    """Deserialization, the tool layer, and random circuits all need this."""
    circuit = Circuit(2).gate("h", 0).gate("cx", 0, 1).gate("rz", 1, params=(0.3,))
    assert [g.name for g in circuit.gates] == ["h", "cx", "rz"]


def test_circuit_gate_validates_like_every_other_builder() -> None:
    with pytest.raises(ValueError, match="unsupported gate"):
        Circuit(2).gate("nope", 0)
    with pytest.raises(ValueError, match="out of range"):
        Circuit(2).gate("h", 9)


def test_circuit_gate_matches_the_named_builders() -> None:
    assert Circuit(2).gate("h", 0).gate("cx", 0, 1) == Circuit(2).h(0).cx(0, 1)


def test_the_private_add_alias_still_works() -> None:
    """17 tests and two library modules used it before it was public."""
    assert Circuit._add is Circuit.gate
    assert Circuit(2)._add("h", 0) == Circuit(2).h(0)


def test_library_code_no_longer_needs_the_private_builder() -> None:
    """Public API for a public need — the point of promoting it."""
    offenders = [
        str(p.relative_to(PACKAGE))
        for p in SOURCES
        if p.name != "ir.py" and "._add(" in p.read_text(encoding="utf-8")
    ]
    assert not offenders, f"still calling the private builder: {offenders}"


# ── 4. every claimed Python version is tested ────────────────────────────


def test_ci_covers_every_python_version_the_classifiers_claim() -> None:
    """A classifier is a support promise; an untested one is a guess."""
    root = PACKAGE.parent.parent
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    workflow = (root / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    claimed = set(re.findall(r'Programming Language :: Python :: (\d+\.\d+)', pyproject))
    tested = set(re.findall(r'"(\d+\.\d+)"', workflow))
    assert claimed, "no Python classifiers found"
    assert claimed <= tested, f"claimed but untested: {sorted(claimed - tested)}"
