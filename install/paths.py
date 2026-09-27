"""Immutable program files and writable per-user data, for source or release."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGED = (ROOT / "release.json").is_file()
DATA = Path.home() / "Library/Application Support/Splash" if PACKAGED else ROOT
MODELS = DATA / "models" if PACKAGED else ROOT / "install/models"
RUNTIME = DATA / "runtime" if PACKAGED else ROOT / "build/runtime"
# Agent homes, with Hermes's sessions and profile: kept out of build/, which
# `make clean` removes, and not in a directory named profiles, which Hermes
# takes for its own layout: it would write its shared files one level up.
AGENTS = RUNTIME if PACKAGED else ROOT / "install/agents"
PYTHON = ROOT / ("python/bin/python3" if PACKAGED else ".venv/bin/python")
BINARY = ROOT / ("engine/splash" if PACKAGED else "build/splash")
