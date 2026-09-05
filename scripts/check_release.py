#!/usr/bin/env python3
"""Dependency-free release checks for the PawCT research repository."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".json", ".html", ".css", ".js", ".txt", ".csv"}
IGNORED_PARTS = {".git", "__pycache__", ".pytest_cache", ".venv"}
MAX_FILE_BYTES = 10 * 1024 * 1024

REQUIRED = (
    "README.md",
    "LICENSE",
    "NOTICE",
    "THIRD_PARTY.md",
    "requirements.txt",
    "src/config.yaml",
    "src/models.py",
    "src/losses.py",
    "docs/index.html",
    "docs/assets/manifest.json",
)

# Build path fragments so this checker does not flag its own source.
FORBIDDEN_TEXT = (
    "/" + "media" + "/",
    "/" + "home" + "/",
    "BEGIN " + "PRIVATE KEY",
    "BEGIN RSA " + "PRIVATE KEY",
    "BEGIN OPENSSH " + "PRIVATE KEY",
)

SECRET_PATTERNS = (
    re.compile(r"(?i)(api[_-]?key|access[_-]?token|client[_-]?secret)\s*[:=]\s*['\"][^'\"]{12,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9_]{30,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
)


def tracked_candidates():
    """Yield tracked and release-candidate files while respecting .gitignore."""
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            check=True,
            capture_output=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        for path in ROOT.rglob("*"):
            if path.is_file() and not any(part in IGNORED_PARTS for part in path.parts):
                yield path
        return

    for relative_bytes in result.stdout.split(b"\0"):
        if not relative_bytes:
            continue
        path = ROOT / relative_bytes.decode("utf-8", errors="surrogateescape")
        if path.is_file():
            yield path


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []

    for relative in REQUIRED:
        if not (ROOT / relative).is_file():
            errors.append(f"missing required file: {relative}")

    for path in tracked_candidates():
        relative = path.relative_to(ROOT)
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            errors.append(f"file exceeds 10 MiB release limit: {relative} ({size} bytes)")

        if path.suffix == ".py":
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=str(relative))
            except (SyntaxError, UnicodeDecodeError) as exc:
                errors.append(f"Python parse failed: {relative}: {exc}")

        if path.suffix not in TEXT_SUFFIXES and path.name not in {"LICENSE", "NOTICE", ".nojekyll"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for fragment in FORBIDDEN_TEXT:
            if fragment in text:
                errors.append(f"private absolute-path/key marker {fragment!r}: {relative}")
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                errors.append(f"possible credential matching {pattern.pattern!r}: {relative}")

    manifest_path = ROOT / "docs/assets/manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for item in manifest.get("assets", []):
                asset = manifest_path.parent / item["path"]
                if not asset.is_file():
                    errors.append(f"manifest asset missing: {asset.relative_to(ROOT)}")
                    continue
                digest = hashlib.sha256(asset.read_bytes()).hexdigest()
                if digest != item.get("sha256"):
                    errors.append(f"manifest SHA-256 mismatch: {asset.relative_to(ROOT)}")
                if not item.get("rights_note"):
                    warnings.append(f"manifest asset lacks rights note: {asset.relative_to(ROOT)}")
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            errors.append(f"invalid asset manifest: {exc}")

    if errors:
        print("Release audit failed:")
        for error in errors:
            print(f"  ERROR: {error}")
    else:
        print("Release audit passed.")
    for warning in warnings:
        print(f"  WARNING: {warning}")
    print(f"Checked repository: {ROOT}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
