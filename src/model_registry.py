"""Discover trusted local model artifacts and archive immutable model versions.

Only local files under the project directories are listed. This module does not
unpickle or execute anything while scanning. Loading happens in inference.py.
"""
from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
KPIS = ("DLPRB", "DLThpVol", "DLThpTime", "MR_number")


@dataclass(frozen=True)
class ModelRecord:
    label: str
    path: str
    family: str
    model: str
    target: str
    version: str
    archived: bool


def _identify(path: Path):
    """Infer metadata from filenames without loading pickle or torch weights."""
    if path.suffix == ".pt":
        sidecar = path.with_suffix(".json")
        if not sidecar.is_file():
            return None
        try:
            config = json.loads(sidecar.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return None
        if config.get("family") != "dl" or config.get("mode") not in (
            "lstm", "gru", "transformer"
        ):
            return None
        return ("dl", str(config["mode"]).upper(), "ALL")
    if path.name == "anomalies.pkl":
        return ("anomaly", "3sigma/IQR/IF/SVM", "ALL")
    if path.suffix == ".pkl":
        for kpi in sorted(KPIS, key=len, reverse=True):
            suffix = "_" + kpi
            if path.stem.endswith(suffix):
                model = path.stem[: -len(suffix)]
                if model:
                    return ("ml", model, kpi)
    return None


def list_models(root: Path | None = None, family: str | None = None) -> list[ModelRecord]:
    """Return all supported versioned and legacy models, newest first."""
    root = Path(root or ROOT)
    resource_root = root / "resources" / "models"
    legacy_root = root / "models"
    records = []
    paths = []
    for directory, archived in ((resource_root, True), (legacy_root, False)):
        if not directory.is_dir():
            continue
        for pattern in ("*.pkl", "*.pt"):
            for path in directory.rglob(pattern):
                if not path.is_file() or path.is_symlink():
                    continue
                if not path.resolve().is_relative_to(directory.resolve()):
                    continue
                paths.append((path, archived))
    for path, archived in paths:
        info = _identify(path)
        if not info:
            continue
        model_family, model_name, target = info
        if family is not None and family != model_family:
            continue
        version = path.parent.name if archived else "legacy"
        label = f"{model_name} | {target} | {version} | {path.name}"
        records.append(ModelRecord(
            label=label, path=str(path.resolve()), family=model_family,
            model=model_name, target=target, version=version, archived=archived
        ))
    return sorted(records, key=lambda item: (Path(item.path).stat().st_mtime, item.label), reverse=True)


def archive_models(
    root: Path | None = None,
    *,
    newer_than: float | None = None,
    reason: str = "manual",
) -> list[ModelRecord]:
    """Copy legacy artifacts to unique version folders without touching originals.

    After a training job, pass its start timestamp to include only files written
    by that job. Existing artifacts can be archived first without newer_than.
    """
    root = Path(root or ROOT)
    origin = root / "models"
    if not origin.is_dir():
        return []
    candidates = []
    for record in list_models(root):
        if record.archived:
            continue
        source = Path(record.path)
        if newer_than is not None and source.stat().st_mtime < newer_than - 2:
            continue
        candidates.append(source)
    if not candidates:
        return []

    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:6]
    folder = root / "resources" / "models" / version
    folder.mkdir(parents=True, exist_ok=False)
    manifest = {"version": version, "reason": reason, "artifacts": []}
    for source in candidates:
        dest = folder / source.name
        shutil.copy2(source, dest)
        if source.suffix == ".pt":
            shutil.copy2(source.with_suffix(".json"), dest.with_suffix(".json"))
        info = _identify(dest)
        manifest["artifacts"].append({
            "filename": dest.name, "family": info[0],
            "model": info[1], "target": info[2],
        })
    (folder / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return [record for record in list_models(root) if record.version == version]
