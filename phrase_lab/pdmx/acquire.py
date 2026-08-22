from __future__ import annotations

import hashlib
import json
import logging
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ZENODO_API = "https://zenodo.org/api/records/{record_id}"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_record(record_id: int) -> dict[str, Any]:
    import requests

    resp = requests.get(ZENODO_API.format(record_id=record_id), timeout=60)
    resp.raise_for_status()
    return resp.json()


def _find_file(record: dict[str, Any], name: str) -> dict[str, Any]:
    for f in record.get("files", []):
        if f.get("key") == name:
            return f
    raise KeyError(f"Missing {name} in record")


def _download(url: str, dest: Path, checksum: str | None = None) -> dict[str, Any]:
    import requests
    from tqdm import tqdm

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    resume = tmp.exists()
    headers = {}
    mode = "ab" if resume else "wb"
    existing = tmp.stat().st_size if resume else 0
    if existing:
        headers["Range"] = f"bytes={existing}-"
    with requests.get(url, stream=True, headers=headers, timeout=60) as r:
        if existing and r.status_code != 206:
            resume = False
            existing = 0
            mode = "wb"
            headers.pop("Range", None)
            r.close()
            with requests.get(url, stream=True, timeout=60) as rr:
                rr.raise_for_status()
                total = int(rr.headers.get("content-length", 0))
                with open(tmp, mode) as f, tqdm(total=total or None, unit="B", unit_scale=True, desc=dest.name) as pbar:
                    for chunk in rr.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            tmp.replace(dest)
            if checksum:
                _verify_checksum(dest, checksum)
            return {"path": str(dest), "size": dest.stat().st_size, "sha256": _sha256(dest)}
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0)) + existing
        with open(tmp, mode) as f, tqdm(total=total or None, initial=existing, unit="B", unit_scale=True, desc=dest.name) as pbar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    tmp.replace(dest)
    if checksum:
        _verify_checksum(dest, checksum)
    return {"path": str(dest), "size": dest.stat().st_size, "sha256": _sha256(dest)}


def _verify_checksum(path: Path, checksum: str) -> None:
    if ":" not in checksum:
        return
    alg, expected = checksum.split(":", 1)
    if alg not in hashlib.algorithms_available:
        return
    h = hashlib.new(alg)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    actual = h.hexdigest()
    if actual != expected:
        raise ValueError(f"Checksum mismatch for {path.name}: {actual} != {expected}")


def _extract_tar(tar_path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(dest)


def download_pdmx(root: str | Path, record_id: int = 15571083, include_mid: bool = False) -> dict[str, Any]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    record = fetch_record(record_id)
    files = {
        "PDMX.csv": _find_file(record, "PDMX.csv"),
        "mxl.tar.gz": _find_file(record, "mxl.tar.gz"),
    }
    if include_mid:
        files["mid.tar.gz"] = _find_file(record, "mid.tar.gz")
    downloaded = {}
    for name, info in files.items():
        dest = root / name
        if not dest.exists():
            downloaded[name] = _download(info["links"]["self"], dest, info.get("checksum"))
        else:
            if info.get("checksum"):
                _verify_checksum(dest, info["checksum"])
            downloaded[name] = {"path": str(dest), "size": dest.stat().st_size, "sha256": _sha256(dest)}
    mxl_tar = root / "mxl.tar.gz"
    mxl_dir = root / "mxl"
    if not mxl_dir.exists() or not any(mxl_dir.iterdir()):
        _extract_tar(mxl_tar, mxl_dir)
    manifest = {
        "record_id": record_id,
        "record_title": record.get("metadata", {}).get("title"),
        "record_version": record.get("metadata", {}).get("version"),
        "record_date": record.get("metadata", {}).get("publication_date"),
        "downloaded": downloaded,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(root / "download_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest
