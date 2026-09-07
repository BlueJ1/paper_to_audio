"""Snapshot installed versions reachable from the declared dependency sets.

Run after validating a deliberate dependency update in a clean Python 3.12
virtual environment with every optional requirements set installed.
"""
from importlib.metadata import distribution
from pathlib import Path
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

root = Path(__file__).resolve().parents[1]
pending = []
for file in root.glob("requirements*.txt"):
    for line in file.read_text().splitlines():
        if line and not line.startswith(("#", "-")):
            pending.append(Requirement(line))
seen, versions = set(), {}
while pending:
    req = pending.pop()
    identity = (canonicalize_name(req.name), tuple(sorted(req.extras)))
    if identity in seen:
        continue
    seen.add(identity)
    package = distribution(req.name)
    versions[canonicalize_name(req.name)] = package.version
    for raw in package.requires or []:
        dep = Requirement(raw)
        if dep.marker is None or any(dep.marker.evaluate({"extra": extra}) for extra in (req.extras or {""})):
            pending.append(dep)
header = "# Python 3.12 installed dependency snapshot, 2026-09-07.\n# Constraints do not install optional packages. Platform-specific dependencies\n# absent on macOS are resolved on their target OS; this is not a hash lock.\n"
(root / "constraints.txt").write_text(header + "".join(f"{key}=={versions[key]}\n" for key in sorted(versions)))
