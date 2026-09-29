#!/usr/bin/env bash
#SBATCH --job-name=muse-runtime
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-muse-runtime-%j.out

set -euo pipefail

ctm_muse_repo="${REPO_DIR:-$PROJECTDIR/$USER/consistency-training-methods}"
cd "$ctm_muse_repo"
export HF_HOME="${HF_HOME:-$SCRATCHDIR/ctm/huggingface-muse}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCHDIR/ctm/uv-cache-muse}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
mkdir -p "$HF_HOME" "$UV_CACHE_DIR"

bash infra/isambard/setup_muse_glimmer_env.sh
source infra/isambard/muse_glimmer_runtime_env.sh

ctm_muse_python="$ctm_muse_repo/.venv-muse/bin/python"
CTM_MUSE_REPO="$ctm_muse_repo" "$ctm_muse_python" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download

repo_id = "meta-models/Muse-Glimmer-30B"
revision = "a4e59da52a7bc87ae7251dd5545c0dd437c44b68"
snapshot = Path(
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
    )
).resolve()
if snapshot.name != revision or snapshot.is_symlink() or not (snapshot / "config.json").is_file():
    raise SystemExit(f"pinned Muse snapshot is absent or malformed: {snapshot}")

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

files = []
for path in sorted(snapshot.iterdir(), key=lambda item: item.name):
    if not path.is_file():
        continue
    resolved = path.resolve()
    blob = resolved.name if len(resolved.name) == 64 and all(c in "0123456789abcdef" for c in resolved.name) else None
    files.append({"name": path.name, "size_bytes": path.stat().st_size, "hf_blob_sha256": blob})
document = {
    "schema": "muse-glimmer-pinned-snapshot-v1",
    "repo_id": repo_id,
    "revision": revision,
    "snapshot_path": str(snapshot),
    "config_sha256": sha256(snapshot / "config.json"),
    "index_sha256": sha256(snapshot / "model.safetensors.index.json"),
    "file_count": len(files),
    "total_bytes": sum(item["size_bytes"] for item in files),
    "files": files,
}
root = Path(os.environ["CTM_MUSE_REPO"]).resolve() / "artifacts" / "muse-glimmer-runtime-20260824"
root.mkdir(parents=True, exist_ok=True)
receipt = root / "model-snapshot.json"
payload = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
if receipt.exists() and receipt.read_bytes() != payload:
    raise SystemExit(f"refusing to overwrite a different Muse snapshot receipt: {receipt}")
receipt.write_bytes(payload)
print(f"CTM_MUSE_SNAPSHOT_RECEIPT={receipt}")
PY
