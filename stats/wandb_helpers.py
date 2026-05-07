"""
wandb_helpers.py — W&B utilities to centralise BESS-Bench monitoring.

Single project: ``anonym-bess-26``.

Usage pattern:

    from stats.wandb_helpers import init_run, log_artifact, finish
    run = init_run(
        project="anonym-bess-26",
        tags=["phase1-coverage"],
        name="coverage_audit_20260423",
        config={"seed": 42, "n_lines": 8},
    )
    run.log({"coverage_Halpha_pct": 98.2})
    log_artifact(run, "results/coverage_audit.json", name="coverage_audit", type="results")
    finish(run)

Safety:
- If ``wandb`` is not installed or ``WANDB_MODE=disabled``, returns a no-op shim.
- Config auto-enriched: git_commit, hostname, python_version, script path.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

PROJECT_DEFAULT = "anonym-bess-26"


class _NoOpRun:
    """Shim used when wandb is unavailable or disabled."""

    id = "noop"

    def log(self, *_args, **_kwargs) -> None:
        return None

    def log_artifact(self, *_args, **_kwargs) -> None:
        return None

    def finish(self) -> None:
        return None

    def use_artifact(self, *_args, **_kwargs) -> None:
        return None

    @property
    def config(self):
        class _Cfg(dict):
            def update(self, *a, **kw):
                return dict.update(self, *a, **kw)
        return _Cfg()


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _enriched_config(extra: Optional[dict]) -> dict:
    cfg = {
        "git_commit": _git_commit(),
        "hostname": socket.gethostname(),
        "python_version": sys.version.split()[0],
        "script": Path(sys.argv[0]).name if sys.argv and sys.argv[0] else "unknown",
        "dataset_revision": "anonym-submit-26/bess-bench-26",
    }
    if extra:
        cfg.update(extra)
    return cfg


def init_run(
    project: str = PROJECT_DEFAULT,
    tags: Optional[Iterable[str]] = None,
    name: Optional[str] = None,
    config: Optional[dict] = None,
    mode: Optional[str] = None,
    group: Optional[str] = None,
    entity: Optional[str] = None,
):
    """Initialise un run W&B. Retourne un shim no-op si wandb absent/disabled."""
    if os.environ.get("WANDB_MODE", "").lower() == "disabled":
        return _NoOpRun()
    try:
        import wandb  # type: ignore
    except ImportError:
        print("[wandb_helpers] wandb not installed, no-op mode", file=sys.stderr)
        return _NoOpRun()

    try:
        run = wandb.init(
            project=project,
            entity=entity,
            name=name,
            tags=list(tags) if tags else None,
            config=_enriched_config(config),
            mode=mode or os.environ.get("WANDB_MODE", "online"),
            group=group,
            reinit=True,
        )
        return run
    except Exception as exc:
        print(f"[wandb_helpers] wandb.init failed ({exc}), falling back to no-op",
              file=sys.stderr)
        return _NoOpRun()


def log_artifact(run, path: str | Path, name: str, type: str = "results",
                 description: str = ""):
    """Log un artifact (fichier unique) sur le run W&B."""
    try:
        import wandb  # type: ignore
    except ImportError:
        return
    path = Path(path)
    if not path.exists():
        print(f"[wandb_helpers] artifact introuvable: {path}", file=sys.stderr)
        return
    try:
        art = wandb.Artifact(name=name, type=type, description=description or name)
        art.add_file(str(path))
        run.log_artifact(art)
    except Exception as exc:
        print(f"[wandb_helpers] log_artifact failed ({exc})", file=sys.stderr)


def finish(run) -> None:
    try:
        run.finish()
    except Exception:
        pass


def add_cli_args(parser) -> None:
    """Add standard W&B flags to an argparse.ArgumentParser."""
    parser.add_argument("--wandb_project", default=PROJECT_DEFAULT,
                        help="W&B project (default: anonym-bess-26)")
    parser.add_argument("--wandb_entity", default=None,
                        help="W&B entity (default: user default)")
    parser.add_argument("--wandb_tags", nargs="*", default=None,
                        help="W&B tags")
    parser.add_argument("--wandb_name", default=None,
                        help="W&B run name (auto-generated if absent)")
    parser.add_argument("--wandb_mode", default=None,
                        choices=[None, "online", "offline", "disabled"],
                        help="W&B mode (default: env WANDB_MODE or online)")
    parser.add_argument("--wandb_group", default=None,
                        help="W&B group (for multi-seed grouping)")
