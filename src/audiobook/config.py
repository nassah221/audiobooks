"""audiobook.toml loading and validation (stdlib tomllib only).

The single, tracked source of truth for every run-scoped input: the book
EPUB, the voice reference pair (audio + transcript), the Qwen model/generation
knobs, and the ASR validation model. All input paths are root-relative; an
absolute or ``..`` escaping path is rejected. Changing any value here changes
the run fingerprint and therefore invalidates resume.
"""
from __future__ import annotations

import os
import pathlib
import tomllib
from dataclasses import dataclass

CONFIG_NAME = "audiobook.toml"


class ConfigError(ValueError):
    """Malformed or incomplete audiobook.toml (caught by the CLI)."""


def find_root(start=None) -> pathlib.Path:
    """Nearest ancestor of `start` (default cwd) containing audiobook.toml."""
    p = pathlib.Path(start or os.getcwd()).resolve()
    for cand in [p, *p.parents]:
        if (cand / CONFIG_NAME).is_file():
            return cand
    raise ConfigError(
        f"no {CONFIG_NAME} at or above {p}; run from the repo or pass --root"
    )


def _section(data, table):
    v = data.get(table)
    if not isinstance(v, dict):
        raise ConfigError(f"[{table}] section missing in {CONFIG_NAME}")
    return v


def _name(data, table, key):
    v = data.get(key)
    if not isinstance(v, str) or not v.strip():
        raise ConfigError(f"[{table}] {key} must be a non-empty string")
    return v.strip()


def _sha(data, table, key):
    v = _name(data, table, key)
    if len(v) != 64 or any(c not in "0123456789abcdef" for c in v):
        raise ConfigError(f"[{table}] {key} must be a 64-char hex sha256")
    return v


def _int(data, table, key):
    v = data.get(key)
    if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
        raise ConfigError(f"[{table}] {key} must be a positive integer")
    return v


def _rel(data, table, key):
    v = _name(data, table, key)
    raw = pathlib.Path(v)
    if raw.is_absolute():
        raise ConfigError(f"[{table}] {key} must be root-relative, got absolute: {v}")
    if ".." in raw.parts:
        raise ConfigError(f"[{table}] {key} must not contain '..': {v}")
    return v


@dataclass(frozen=True)
class Config:
    root: pathlib.Path
    book: str
    book_sha256: str
    audio: str
    audio_sha256: str
    transcript: str
    transcript_sha256: str
    model_repo: str
    model_revision: str
    language: str
    max_tokens: int
    seed: int
    asr_repo: str
    asr_revision: str

    @property
    def inputs(self) -> dict:
        """{relative_input_path: expected_sha256} consumed by verify_inputs."""
        return {
            self.book: self.book_sha256,
            self.audio: self.audio_sha256,
            self.transcript: self.transcript_sha256,
        }

    def abs(self, rel: str) -> pathlib.Path:
        return self.root / rel


def load_config(root) -> Config:
    """Read and validate audiobook.toml under `root` -> Config (paths relative)."""
    root = pathlib.Path(root)
    p = root / CONFIG_NAME
    if not p.is_file():
        raise ConfigError(f"no {CONFIG_NAME} at {root}")
    with open(p, "rb") as f:
        data = tomllib.load(f)
    book = _section(data, "book")
    voice = _section(data, "voice")
    model = _section(data, "model")
    asr_ = _section(data, "asr")
    return Config(
        root=root,
        book=_rel(book, "book", "path"),
        book_sha256=_sha(book, "book", "sha256"),
        audio=_rel(voice, "voice", "audio"),
        audio_sha256=_sha(voice, "voice", "audio_sha256"),
        transcript=_rel(voice, "voice", "transcript"),
        transcript_sha256=_sha(voice, "voice", "transcript_sha256"),
        model_repo=_name(model, "model", "repo"),
        model_revision=_name(model, "model", "revision"),
        language=_name(model, "model", "language"),
        max_tokens=_int(model, "model", "max_tokens"),
        seed=_int(model, "model", "seed"),
        asr_repo=_name(asr_, "asr", "repo"),
        asr_revision=_name(asr_, "asr", "revision"),
    )
