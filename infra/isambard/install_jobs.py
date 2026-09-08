#!/usr/bin/env python3
"""Install independent controller copies, preserving previous versions."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

MODULES = ("jobs.py", "job_controller.py", "job_transport.py", "job_adapters.py", "JOBS.md", "AGENTS.jobs.md")


def install(prefix):
    source = Path(__file__).resolve().parent
    for name in MODULES:
        if not (source / name).is_file():
            raise RuntimeError("Missing installation input: " + name)
    destination = prefix / "share/ctm-isambard"
    bin_dir = prefix / "bin"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archive = destination / "_archive" / (stamp + "-jobs")
    for directory in (destination, bin_dir):
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    copies = [(source / name, destination / name, 0o600) for name in MODULES]
    launcher = (
        "#!/usr/bin/env python3\n"
        "import runpy, sys\n"
        "sys.path.insert(0, " + repr(str(destination)) + ")\n"
        "runpy.run_path(" + repr(str(destination / "jobs.py")) + ", run_name='__main__')\n"
    )
    changed = []
    for src, target, mode in copies + [(None, bin_dir / "isambard-jobs", 0o755)]:
        body = src.read_bytes() if src else launcher.encode()
        if target.is_symlink():
            raise RuntimeError("Refusing to replace a symlink: " + str(target))
        if target.exists() and target.read_bytes() == body:
            continue
        if target.exists():
            archive.mkdir(mode=0o700, parents=True, exist_ok=True)
            shutil.copy2(target, archive / target.name)
        fd, temporary = tempfile.mkstemp(prefix=".install-jobs-", dir=str(target.parent))
        with os.fdopen(fd, "wb") as stream:
            os.fchmod(stream.fileno(), mode)
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        changed.append(str(target))
    return {
        "installed": changed,
        "archive": str(archive) if archive.exists() else None,
        "command": str(bin_dir / "isambard-jobs"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, default=Path.home() / ".local")
    args = parser.parse_args()
    print(json.dumps(install(args.prefix), indent=2))
