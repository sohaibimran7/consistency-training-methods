#!/usr/bin/env bash
#SBATCH --job-name=muse-runtime-cu129
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-muse-runtime-cu129-%j.out

set -euo pipefail

ctm_muse_repo="${REPO_DIR:-$PROJECTDIR/$USER/consistency-training-methods}"
cd "$ctm_muse_repo"
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-$SCRATCHDIR/ctm/huggingface-muse}"
export CTM_MUSE_RUNTIME_RECEIPT_DIR="$ctm_muse_repo/artifacts/muse-glimmer-runtime-cu129-20260824"
ctm_muse_job_tmp="$SCRATCHDIR/ctm/tmp/muse-runtime-cu129-$SLURM_JOB_ID"
mkdir -p "$ctm_muse_job_tmp"
export TMPDIR="$ctm_muse_job_tmp"
export TMP="$ctm_muse_job_tmp"
export TEMP="$ctm_muse_job_tmp"
source infra/isambard/muse_glimmer_cuda129_runtime_env.sh

if [[ -e "$CTM_MUSE_RUNTIME_RECEIPT_DIR" || -L "$CTM_MUSE_RUNTIME_RECEIPT_DIR" ]]; then
    echo "ERROR: refusing to overwrite source-runtime receipt: $CTM_MUSE_RUNTIME_RECEIPT_DIR" >&2
    exit 2
fi

CTM_MUSE_REPO="$ctm_muse_repo" \
CTM_MUSE_PYTHON="$CTM_MUSE_RUNTIME_PYTHON" \
"$CTM_MUSE_RUNTIME_PYTHON" - <<'PY'
import ctypes
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlparse

repo = Path(os.environ["CTM_MUSE_REPO"]).resolve()
python = Path(os.environ["CTM_MUSE_PYTHON"])
cuda_home = Path(os.environ["CTM_MUSE_CUDA_HOME"]).resolve()
source = Path(os.environ["CTM_MUSE_VLLM_SOURCE"]).resolve()
output = Path(os.environ["CTM_MUSE_RUNTIME_RECEIPT_DIR"]).resolve()
partial = output.parent / f".{output.name}.job-{os.environ['SLURM_JOB_ID']}.partial"
expected_commit = "8c2bbe00d58a930c6c09a80495728b26b79d9200"
expected_version = "0.26.1rc1.dev1136+g8c2bbe00d"
installer = Path(os.environ["SCRATCHDIR"]) / "ctm/cuda-installers/cuda_12.9.1_575.57.08_linux_sbsa.run"
snapshot_receipt_source = repo / "artifacts/muse-glimmer-runtime-20260824/model-snapshot.json"

if not python.is_absolute() or not python.is_file() or not os.access(python, os.X_OK):
    raise SystemExit("Muse venv Python spelling is absent or not executable")

def run(*args: str) -> str:
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def identity(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    if path.is_symlink() or not resolved.is_file():
        raise SystemExit(f"required regular file is absent: {path}")
    return {"path": str(resolved), "sha256": sha256(resolved), "size_bytes": resolved.stat().st_size}

if partial.exists() or partial.is_symlink():
    raise SystemExit(f"refusing to reuse partial runtime receipt: {partial}")
partial.mkdir(parents=True, exist_ok=False)

commit = run("git", "-C", str(source), "rev-parse", "HEAD").strip()
tree = run("git", "-C", str(source), "rev-parse", "HEAD^{tree}").strip()
status = run("git", "-C", str(source), "status", "--porcelain")
if commit != expected_commit or status:
    raise SystemExit("vLLM source checkout is not the clean pinned commit")

import torch
import transformers
import vllm

if importlib.metadata.version("vllm") != expected_version:
    raise SystemExit("vLLM package version changed")
if not Path(vllm.__file__).resolve().is_relative_to(source):
    raise SystemExit("vLLM import does not resolve to the pinned checkout")
distribution = importlib.metadata.distribution("vllm")
direct_url_text = distribution.read_text("direct_url.json")
if not direct_url_text:
    raise SystemExit("editable vLLM installation lacks PEP-610 metadata")
direct_url = json.loads(direct_url_text)
if direct_url.get("dir_info", {}).get("editable") is not True:
    raise SystemExit("vLLM source installation is not recorded as editable")
url = direct_url.get("url")
if not isinstance(url, str) or urlparse(url).scheme != "file" \
        or Path(unquote(urlparse(url).path)).resolve() != source:
    raise SystemExit("vLLM PEP-610 source path differs from the pinned checkout")

extensions = sorted(Path(vllm.__file__).resolve().parent.rglob("*.so"))
if not extensions:
    raise SystemExit("vLLM source runtime has no native extensions")
extension_records = []
saw_cuda12 = False
for extension in extensions:
    linked = run("ldd", str(extension))
    if "not found" in linked or "libcudart.so.13" in linked:
        raise SystemExit(f"invalid native linkage for {extension}:\n{linked}")
    saw_cuda12 = saw_cuda12 or "libcudart.so.12" in linked
    record = identity(extension)
    record["relative_path"] = str(extension.relative_to(source))
    record["ldd"] = linked.splitlines()
    extension_records.append(record)
if not saw_cuda12:
    raise SystemExit("no vLLM extension linked the required CUDA-12 runtime")

dependency_records = []
for dependency in sorted((source / ".deps").glob("*-src")):
    if not (dependency / ".git").exists():
        continue
    dependency_status = run("git", "-C", str(dependency), "status", "--porcelain")
    if dependency_status:
        raise SystemExit(f"vLLM native dependency checkout is dirty: {dependency}")
    submodules = run("git", "-C", str(dependency), "submodule", "status", "--recursive").splitlines()
    if any(line.startswith(("-", "+", "U")) for line in submodules):
        raise SystemExit(f"vLLM native dependency has an unpinned submodule: {dependency}")
    dependency_records.append(
        {
            "name": dependency.name,
            "path": str(dependency.resolve()),
            "commit": run("git", "-C", str(dependency), "rev-parse", "HEAD").strip(),
            "tree": run("git", "-C", str(dependency), "rev-parse", "HEAD^{tree}").strip(),
            "source_clean": True,
            "submodules": submodules,
        }
    )
if not dependency_records:
    raise SystemExit("vLLM source runtime has no pinned native dependency checkouts")

if torch.version.cuda != "12.9" or not torch.cuda.is_available():
    raise SystemExit("torch is not using its pinned CUDA-12.9 runtime")
driver = ctypes.CDLL("libcuda.so.1")
if driver.cuInit(0) != 0:
    raise SystemExit("CUDA driver initialization failed")
driver_api = ctypes.c_int()
if driver.cuDriverGetVersion(ctypes.byref(driver_api)) != 0 or driver_api.value < 12000:
    raise SystemExit("CUDA driver API is incompatible with the CUDA-12 runtime")

if installer.stat().st_size != 5_074_335_372:
    raise SystemExit("CUDA installer byte size changed")
if md5(installer) != "824d4aea22be39358bb809b4f738488b":
    raise SystemExit("CUDA installer NVIDIA MD5 changed")
if sha256(installer) != "64f47ab791a76b6889702425e0755385f5fa216c5a9f061875c7deed5f08cdb6":
    raise SystemExit("CUDA installer SHA-256 changed")
nvcc = run(str(cuda_home / "bin/nvcc"), "--version")
if "release 12.9, V12.9.86" not in nvcc:
    raise SystemExit("CUDA toolkit compiler version changed")

freeze = run("uv", "pip", "freeze", "--python", str(python))
freeze_path = partial / "pip-freeze.txt"
freeze_path.write_text(freeze, encoding="utf-8")
snapshot = json.loads(snapshot_receipt_source.read_text(encoding="utf-8"))
if snapshot.get("revision") != "a4e59da52a7bc87ae7251dd5545c0dd437c44b68":
    raise SystemExit("source model-snapshot receipt has the wrong revision")
snapshot_path = partial / "model-snapshot.json"
snapshot_path.write_bytes(snapshot_receipt_source.read_bytes())

final_freeze = output / "pip-freeze.txt"
document = {
    "schema": "muse-glimmer-isambard-source-runtime-v1",
    "platform": platform.platform(),
    "python": platform.python_version(),
    "python_executable": str(python),
    "python_base_executable": str(python.resolve()),
    "generation_policy": {"max_tokens": None, "termination": "eos_only"},
    "driver": {
        "cuda_driver_api": driver_api.value,
        "nvidia_smi": run("nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader").splitlines(),
    },
    "cuda_toolkit": {
        "home": str(cuda_home),
        "nvcc_version": nvcc.splitlines(),
        "installer": {
            **identity(installer),
            "official_size_bytes": 5_074_335_372,
            "official_md5": "824d4aea22be39358bb809b4f738488b",
            "sha256": "64f47ab791a76b6889702425e0755385f5fa216c5a9f061875c7deed5f08cdb6",
        },
    },
    "torch": {"version": importlib.metadata.version("torch"), "cuda": torch.version.cuda},
    "transformers": transformers.__version__,
    "vllm": {
        "version": importlib.metadata.version("vllm"),
        "commit": commit,
        "tree": tree,
        "source_path": str(source),
        "source_clean": True,
        "direct_url": direct_url,
        "extensions": extension_records,
        "native_dependencies": dependency_records,
    },
    "pip_freeze": {
        "path": str(final_freeze),
        "sha256": sha256(freeze_path),
    },
    "build_inputs": {
        name: identity(repo / "infra/isambard" / name)
        for name in (
            "build_muse_glimmer_vllm_cuda129_source.sbatch",
            "retry_muse_glimmer_vllm_cuda129_source.sbatch",
            "muse_glimmer_cuda129_runtime_env.sh",
            "attest_muse_glimmer_cuda129_source_runtime.sbatch",
            "setup_muse_glimmer_env.sh",
            "muse-runtime-requirements.txt",
        )
    },
    "source_model_snapshot_receipt": identity(snapshot_receipt_source),
}
(partial / "runtime.json").write_text(
    json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
    encoding="utf-8",
)
partial.rename(output)
print(f"CTM_MUSE_SOURCE_RUNTIME_RECEIPT={output / 'runtime.json'}")
print(f"CTM_MUSE_SOURCE_RUNTIME_FREEZE={output / 'pip-freeze.txt'}")
print(f"CTM_MUSE_SOURCE_RUNTIME_MODEL={output / 'model-snapshot.json'}")
PY
