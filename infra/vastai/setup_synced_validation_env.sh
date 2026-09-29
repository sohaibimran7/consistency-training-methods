#!/usr/bin/env bash
# Build an isolated validation environment around a vLLM image's CUDA/Torch stack.
set -euo pipefail

CTM_VALIDATION_ROOT="${CTM_VALIDATION_ROOT:-/workspace/ctm-qwen-validation}"
CTM_VALIDATION_REPO="${CTM_VALIDATION_REPO:-$CTM_VALIDATION_ROOT/repo}"
CTM_VALIDATION_ENV="${CTM_VALIDATION_ENV:-$CTM_VALIDATION_ROOT/env}"

# `causal-conv1d` 1.6.2.post1's setup.py appends a broad fixed CUDA matrix
# itself (sm_75, sm_80, sm_87, sm_90, and newer targets on newer toolkits).
# Consequently, merely exporting TORCH_CUDA_ARCH_LIST does *not* limit its
# local build.  The values below identify the upstream source archive and its
# setup.py that this script is allowed to patch.  Do not make this a floating
# package upgrade: the patch is deliberately fail-closed if the source changes.
CTM_CAUSAL_CONV1D_DEFAULT_VERSION="1.6.2.post1"
CTM_CAUSAL_CONV1D_DEFAULT_SHA256="245e314ea21064ded7a5bf6b3b842b644aa6f92e45cecfe3e935629744c35ff4"
CTM_CAUSAL_CONV1D_DEFAULT_SETUP_SHA256="0447ac5c9769f8ef8e94a6d57d8c65d81c2786a3ef72f2c107d07814d62328b3"
CTM_CAUSAL_CONV1D_PATCH_ID="ctm-causal-conv1d-arch-list-v1"

CTM_CAUSAL_CONV1D_VERSION="${CTM_CAUSAL_CONV1D_VERSION:-$CTM_CAUSAL_CONV1D_DEFAULT_VERSION}"
if [[ -z "${CTM_CAUSAL_CONV1D_SHA256+x}" ]]; then
    CTM_CAUSAL_CONV1D_SHA256="$CTM_CAUSAL_CONV1D_DEFAULT_SHA256"
    CTM_CAUSAL_CONV1D_SHA256_WAS_DEFAULT=1
else
    CTM_CAUSAL_CONV1D_SHA256_WAS_DEFAULT=0
fi
if [[ -z "${CTM_CAUSAL_CONV1D_SETUP_SHA256+x}" ]]; then
    CTM_CAUSAL_CONV1D_SETUP_SHA256="$CTM_CAUSAL_CONV1D_DEFAULT_SETUP_SHA256"
    CTM_CAUSAL_CONV1D_SETUP_SHA256_WAS_DEFAULT=1
else
    CTM_CAUSAL_CONV1D_SETUP_SHA256_WAS_DEFAULT=0
fi

# CTM_CAUSAL_CONV1D_ARCH_LIST is the explicit preferred override.  Retaining
# TORCH_CUDA_ARCH_LIST as a fallback keeps existing launch commands working.
# H100 and H200 are compute capability 9.0, so this produces sm_90 only.
CTM_CAUSAL_CONV1D_ARCH_LIST="${CTM_CAUSAL_CONV1D_ARCH_LIST:-${TORCH_CUDA_ARCH_LIST:-9.0}}"
CTM_CAUSAL_CONV1D_VERIFY_ARCHES="${CTM_CAUSAL_CONV1D_VERIFY_ARCHES:-1}"

ctm_fail() {
    echo "ERROR: $*" >&2
    exit 2
}

if [[ ! "$CTM_CAUSAL_CONV1D_VERSION" =~ ^[0-9A-Za-z][0-9A-Za-z._+-]*$ ]]; then
    ctm_fail "CTM_CAUSAL_CONV1D_VERSION must be a simple package version."
fi
if [[ ! "$CTM_CAUSAL_CONV1D_SHA256" =~ ^[0-9a-fA-F]{64}$ ]]; then
    ctm_fail "CTM_CAUSAL_CONV1D_SHA256 must be a 64-character SHA-256 digest."
fi
if [[ ! "$CTM_CAUSAL_CONV1D_SETUP_SHA256" =~ ^[0-9a-fA-F]{64}$ ]]; then
    ctm_fail "CTM_CAUSAL_CONV1D_SETUP_SHA256 must be a 64-character SHA-256 digest."
fi
# Keep the launcher usable with the Bash 3.x shipped on older controllers;
# the actual target environment is Linux, but no Bash-4-only lowercase syntax
# is needed here.
CTM_CAUSAL_CONV1D_SHA256="$(printf '%s' "$CTM_CAUSAL_CONV1D_SHA256" | tr '[:upper:]' '[:lower:]')"
CTM_CAUSAL_CONV1D_SETUP_SHA256="$(printf '%s' "$CTM_CAUSAL_CONV1D_SETUP_SHA256" | tr '[:upper:]' '[:lower:]')"

if [[ "$CTM_CAUSAL_CONV1D_VERSION" != "$CTM_CAUSAL_CONV1D_DEFAULT_VERSION" ]] && \
    [[ "$CTM_CAUSAL_CONV1D_SHA256_WAS_DEFAULT" == "1" || "$CTM_CAUSAL_CONV1D_SETUP_SHA256_WAS_DEFAULT" == "1" ]]; then
    ctm_fail "a custom causal-conv1d version requires both CTM_CAUSAL_CONV1D_SHA256 and CTM_CAUSAL_CONV1D_SETUP_SHA256."
fi

CTM_CAUSAL_CONV1D_ARCH_REGEX='^[0-9]+([.][0-9]+)?([+]PTX)?([[:space:];]+[0-9]+([.][0-9]+)?([+]PTX)?)*$'
if ! [[ "$CTM_CAUSAL_CONV1D_ARCH_LIST" =~ $CTM_CAUSAL_CONV1D_ARCH_REGEX ]]; then
    ctm_fail "CTM_CAUSAL_CONV1D_ARCH_LIST must be a numeric Torch CUDA arch list such as '9.0' or '8.0;9.0+PTX'."
fi
case "$CTM_CAUSAL_CONV1D_VERIFY_ARCHES" in
    0|1) ;;
    *) ctm_fail "CTM_CAUSAL_CONV1D_VERIFY_ARCHES must be 0 or 1." ;;
esac
echo "causal-conv1d source-build plan: version=$CTM_CAUSAL_CONV1D_VERSION sha256=$CTM_CAUSAL_CONV1D_SHA256 target_arches=$CTM_CAUSAL_CONV1D_ARCH_LIST"
echo "  Override target_arches with CTM_CAUSAL_CONV1D_ARCH_LIST (or legacy TORCH_CUDA_ARCH_LIST); a custom version requires both source and setup.py SHA-256 overrides."

if [[ ! -f "$CTM_VALIDATION_REPO/requirements.txt" ]]; then
    echo "ERROR: synced repository is missing at $CTM_VALIDATION_REPO" >&2
    exit 2
fi

if [[ ! -x "$CTM_VALIDATION_ENV/bin/python" ]]; then
    python3 -m venv --system-site-packages "$CTM_VALIDATION_ENV"
fi

CTM_VALIDATION_PYTHON="$CTM_VALIDATION_ENV/bin/python"
"$CTM_VALIDATION_PYTHON" -m pip install --upgrade pip 'setuptools>=77,<81' wheel packaging ninja
"$CTM_VALIDATION_PYTHON" -m pip install -r "$CTM_VALIDATION_REPO/requirements.txt"
"$CTM_VALIDATION_PYTHON" -m pip install --no-deps -e "$CTM_VALIDATION_REPO"

# Qwen3.5's hybrid DeltaNet layers are unusably slow through the Torch fallback.
# Build/import these only after inheriting the image's resolved CUDA/Torch stack.
# H100 and H200 are both compute capability 9.0. The causal-conv1d source is
# patched below before installation so this exported value controls the actual
# nvcc gencode list, rather than being shadowed by upstream hard-coded flags.
export TORCH_CUDA_ARCH_LIST="$CTM_CAUSAL_CONV1D_ARCH_LIST"
"$CTM_VALIDATION_PYTHON" -m pip install --upgrade flash-linear-attention

CTM_CAUSAL_CONV1D_CUDA_HOME="$("$CTM_VALIDATION_PYTHON" - <<'PY'
from torch.utils.cpp_extension import CUDA_HOME

print(CUDA_HOME or "")
PY
)"
if [[ -z "$CTM_CAUSAL_CONV1D_CUDA_HOME" || ! -f "$CTM_CAUSAL_CONV1D_CUDA_HOME/include/cuda.h" ]]; then
    ctm_fail "causal-conv1d requires CUDA development headers; the inherited image is runtime-only."
fi
if [[ ! -x "$CTM_CAUSAL_CONV1D_CUDA_HOME/bin/nvcc" ]]; then
    ctm_fail "causal-conv1d requires nvcc at $CTM_CAUSAL_CONV1D_CUDA_HOME/bin/nvcc."
fi

# Use an immutable archive cache and a target-specific source tree.  A source
# tree includes the arch-list digest, so changing the requested architecture
# cannot accidentally reuse objects built for a previous target.  Nothing is
# deleted on rerun; a partial or unexpected tree fails closed for inspection.
CTM_CAUSAL_CONV1D_CACHE_DIR="$CTM_VALIDATION_ROOT/kernel-sources/causal-conv1d/$CTM_CAUSAL_CONV1D_VERSION/$CTM_CAUSAL_CONV1D_SHA256"
CTM_CAUSAL_CONV1D_SOURCE_PARENT="$CTM_VALIDATION_ROOT/kernel-sources/causal-conv1d/builds"
mkdir -p "$CTM_CAUSAL_CONV1D_CACHE_DIR" "$CTM_CAUSAL_CONV1D_SOURCE_PARENT"

CTM_CAUSAL_CONV1D_ARCH_ID="$("$CTM_VALIDATION_PYTHON" - "$CTM_CAUSAL_CONV1D_ARCH_LIST" <<'PY'
import hashlib
import sys

print(hashlib.sha256(sys.argv[1].encode("utf-8")).hexdigest()[:12])
PY
)"

CTM_CAUSAL_CONV1D_ARCHIVE="$("$CTM_VALIDATION_PYTHON" - "$CTM_CAUSAL_CONV1D_CACHE_DIR" <<'PY'
from pathlib import Path
import sys

cache = Path(sys.argv[1])
archives = sorted(path for path in cache.iterdir() if path.is_file() and path.name.endswith(".tar.gz"))
if len(archives) > 1:
    raise SystemExit(f"expected at most one causal-conv1d source archive in {cache}, found {archives}")
if archives:
    print(archives[0])
PY
)"
if [[ -z "$CTM_CAUSAL_CONV1D_ARCHIVE" ]]; then
    "$CTM_VALIDATION_PYTHON" -m pip download \
        --disable-pip-version-check \
        --no-deps \
        --no-binary=causal-conv1d \
        --dest "$CTM_CAUSAL_CONV1D_CACHE_DIR" \
        "causal-conv1d==$CTM_CAUSAL_CONV1D_VERSION"
    CTM_CAUSAL_CONV1D_ARCHIVE="$("$CTM_VALIDATION_PYTHON" - "$CTM_CAUSAL_CONV1D_CACHE_DIR" <<'PY'
from pathlib import Path
import sys

cache = Path(sys.argv[1])
archives = sorted(path for path in cache.iterdir() if path.is_file() and path.name.endswith(".tar.gz"))
if len(archives) != 1:
    raise SystemExit(f"expected exactly one causal-conv1d source archive in {cache}, found {archives}")
print(archives[0])
PY
)"
fi

CTM_CAUSAL_CONV1D_ACTUAL_SHA256="$("$CTM_VALIDATION_PYTHON" - "$CTM_CAUSAL_CONV1D_ARCHIVE" <<'PY'
import hashlib
from pathlib import Path
import sys

digest = hashlib.sha256()
with Path(sys.argv[1]).open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
print(digest.hexdigest())
PY
)"
if [[ "$CTM_CAUSAL_CONV1D_ACTUAL_SHA256" != "$CTM_CAUSAL_CONV1D_SHA256" ]]; then
    ctm_fail "causal-conv1d source SHA-256 mismatch: expected $CTM_CAUSAL_CONV1D_SHA256, got $CTM_CAUSAL_CONV1D_ACTUAL_SHA256."
fi

CTM_CAUSAL_CONV1D_SOURCE_ROOT="$CTM_CAUSAL_CONV1D_SOURCE_PARENT/${CTM_CAUSAL_CONV1D_VERSION}-${CTM_CAUSAL_CONV1D_SHA256:0:12}-${CTM_CAUSAL_CONV1D_ARCH_ID}"
CTM_CAUSAL_CONV1D_SOURCE_DIR="$("$CTM_VALIDATION_PYTHON" - "$CTM_CAUSAL_CONV1D_ARCHIVE" "$CTM_CAUSAL_CONV1D_SOURCE_ROOT" <<'PY'
from pathlib import Path, PurePosixPath
import os
import sys
import tarfile
import tempfile

archive = Path(sys.argv[1])
destination = Path(sys.argv[2])

def source_dir(root: Path) -> Path:
    candidates = sorted(path.parent for path in root.glob("*/setup.py") if path.parent.is_dir())
    if len(candidates) != 1:
        raise RuntimeError(f"expected exactly one source directory with setup.py under {root}, found {candidates}")
    source = candidates[0]
    required = (
        source / "causal_conv1d" / "__init__.py",
        source / "csrc" / "causal_conv1d.cpp",
        source / "csrc" / "causal_conv1d_fwd.cu",
        source / "csrc" / "causal_conv1d_bwd.cu",
        source / "csrc" / "causal_conv1d_update.cu",
    )
    missing = [str(path.relative_to(source)) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"incomplete causal-conv1d source tree under {source}: missing {missing}")
    return source

if destination.exists():
    print(source_dir(destination))
else:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.extract-", dir=destination.parent))
    try:
        with tarfile.open(archive, "r:gz") as tar:
            members = tar.getmembers()
            for member in members:
                name = PurePosixPath(member.name)
                if (
                    name.is_absolute()
                    or ".." in name.parts
                    or member.issym()
                    or member.islnk()
                    or not (member.isfile() or member.isdir())
                ):
                    raise RuntimeError(f"unsafe member in causal-conv1d archive: {member.name!r}")
            # Python 3.9 is still used by some vLLM images and has no
            # tarfile extraction-filter argument.  The explicit validation
            # above rejects paths, links, and special archive members before
            # this compatibility extraction call.
            tar.extractall(temporary, members=members)
        source_dir(temporary)
        os.rename(temporary, destination)
    except Exception:
        # Preserve the temporary directory for inspection rather than deleting it.
        raise
    print(source_dir(destination))
PY
)"

# This narrowly removes only the fixed gencode block from the SHA-verified
# upstream setup.py.  PyTorch's BuildExtension then derives gencode flags from
# TORCH_CUDA_ARCH_LIST (9.0 -> arch=compute_90,code=sm_90).  The package's
# CAUSAL_CONV1D_FORCE_BUILD escape hatch prevents its wheel downloader from
# replacing this patched source build with an upstream prebuilt wheel.
CTM_CAUSAL_CONV1D_PATCHED_SETUP_SHA256="$("$CTM_VALIDATION_PYTHON" - \
    "$CTM_CAUSAL_CONV1D_SOURCE_DIR/setup.py" \
    "$CTM_CAUSAL_CONV1D_SETUP_SHA256" \
    "$CTM_CAUSAL_CONV1D_PATCH_ID" <<'PY'
import hashlib
from pathlib import Path
import sys

setup_path = Path(sys.argv[1])
expected_sha256 = sys.argv[2]
patch_id = sys.argv[3]
text = setup_path.read_text(encoding="utf-8")
current_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
marker = "CTM_CAUSAL_CONV1D_ARCH_PATCH_V1"
start = "        cc_flag.append(\"-gencode\")\n        cc_flag.append(\"arch=compute_75,code=sm_75\")"
end = "            cc_flag.append(\"arch=compute_121,code=sm_121\")\n"

if current_sha256 == expected_sha256:
    start_index = text.find(start)
    end_index = text.find(end, start_index)
    if start_index < 0 or end_index < 0:
        raise SystemExit("verified causal-conv1d setup.py does not contain the expected hard-coded CUDA architecture block")
    end_index += len(end)
    replacement = (
        f"        # {marker}: {patch_id}\n"
        "        # Let torch.utils.cpp_extension.BuildExtension derive every CUDA target\n"
        "        # from TORCH_CUDA_ARCH_LIST instead of compiling upstream's broad matrix.\n"
    )
    text = text[:start_index] + replacement + text[end_index:]
    setup_path.write_text(text, encoding="utf-8")
    print("patched verified causal-conv1d setup.py", file=sys.stderr)
elif marker in text:
    forbidden = (
        "arch=compute_75,code=sm_75",
        "arch=compute_80,code=sm_80",
        "arch=compute_87,code=sm_87",
        "arch=compute_90,code=sm_90",
        "arch=compute_100,code=sm_100",
        "arch=compute_120,code=sm_120",
        "arch=compute_103,code=sm_103",
        "arch=compute_110,code=sm_110",
        "arch=compute_121,code=sm_121",
    )
    if any(item in text for item in forbidden):
        raise SystemExit("causal-conv1d source tree has a partial architecture patch; preserve it and inspect before retrying")
    print("reusing previously patched causal-conv1d source tree", file=sys.stderr)
else:
    raise SystemExit(
        f"causal-conv1d setup.py SHA-256 mismatch: expected upstream {expected_sha256}, got {current_sha256}; "
        "refusing to patch an unrecognised source tree"
    )

print(hashlib.sha256(setup_path.read_bytes()).hexdigest())
PY
)"

export CAUSAL_CONV1D_FORCE_BUILD=TRUE
"$CTM_VALIDATION_PYTHON" -m pip install \
    --disable-pip-version-check \
    --no-cache-dir \
    --no-deps \
    --no-build-isolation \
    --force-reinstall \
    "$CTM_CAUSAL_CONV1D_SOURCE_DIR"

CTM_CAUSAL_CONV1D_INSTALLED_VERSION="$("$CTM_VALIDATION_PYTHON" - <<'PY'
from importlib.metadata import version

print(version("causal-conv1d"))
PY
)"
CTM_CAUSAL_CONV1D_PROVENANCE_DIR="$CTM_VALIDATION_ROOT/provenance"
mkdir -p "$CTM_CAUSAL_CONV1D_PROVENANCE_DIR"
CTM_CAUSAL_CONV1D_STAMP="$(date -u +%Y%m%dT%H%M%SZ)-$$"
CTM_CAUSAL_CONV1D_PROVENANCE_FILE="$CTM_CAUSAL_CONV1D_PROVENANCE_DIR/causal-conv1d-${CTM_CAUSAL_CONV1D_VERSION}-${CTM_CAUSAL_CONV1D_SHA256:0:12}-${CTM_CAUSAL_CONV1D_ARCH_ID}-${CTM_CAUSAL_CONV1D_STAMP}.json"
CTM_CAUSAL_CONV1D_FATBIN_FILE=""
CTM_CAUSAL_CONV1D_FOUND_ARCHES=""
CTM_CAUSAL_CONV1D_VERIFICATION_STATUS="disabled"

if [[ "$CTM_CAUSAL_CONV1D_VERIFY_ARCHES" == "1" ]]; then
    if [[ -x "$CTM_CAUSAL_CONV1D_CUDA_HOME/bin/cuobjdump" ]]; then
        CTM_CAUSAL_CONV1D_CUOBJDUMP="$CTM_CAUSAL_CONV1D_CUDA_HOME/bin/cuobjdump"
    elif command -v cuobjdump >/dev/null 2>&1; then
        CTM_CAUSAL_CONV1D_CUOBJDUMP="$(command -v cuobjdump)"
    else
        ctm_fail "cannot verify causal-conv1d CUDA targets because cuobjdump is unavailable; set CTM_CAUSAL_CONV1D_VERIFY_ARCHES=0 only with an explicit review."
    fi
    CTM_CAUSAL_CONV1D_EXTENSION="$("$CTM_VALIDATION_PYTHON" - <<'PY'
import importlib.util

spec = importlib.util.find_spec("causal_conv1d_cuda")
if spec is None or spec.origin is None:
    raise SystemExit("causal_conv1d_cuda extension was not installed")
print(spec.origin)
PY
)"
    CTM_CAUSAL_CONV1D_FATBIN_FILE="$CTM_CAUSAL_CONV1D_PROVENANCE_DIR/causal-conv1d-${CTM_CAUSAL_CONV1D_VERSION}-${CTM_CAUSAL_CONV1D_SHA256:0:12}-${CTM_CAUSAL_CONV1D_ARCH_ID}-${CTM_CAUSAL_CONV1D_STAMP}.fatbin.txt"
    {
        "$CTM_CAUSAL_CONV1D_CUOBJDUMP" --list-elf "$CTM_CAUSAL_CONV1D_EXTENSION"
        printf '\n--- PTX ---\n'
        "$CTM_CAUSAL_CONV1D_CUOBJDUMP" --list-ptx "$CTM_CAUSAL_CONV1D_EXTENSION" || true
    } > "$CTM_CAUSAL_CONV1D_FATBIN_FILE"
    CTM_CAUSAL_CONV1D_FOUND_ARCHES="$("$CTM_VALIDATION_PYTHON" - \
        "$CTM_CAUSAL_CONV1D_ARCH_LIST" \
        "$CTM_CAUSAL_CONV1D_FATBIN_FILE" <<'PY'
import re
from pathlib import Path
import sys

requested = {
    part.removesuffix("+PTX").replace(".", "")
    for part in re.split(r"[;\s]+", sys.argv[1].strip())
    if part
}
contents = Path(sys.argv[2]).read_text(encoding="utf-8", errors="replace")
found = set(re.findall(r"\b(?:sm|compute)_([0-9]+[a-z]?)\b", contents))
if not found:
    raise SystemExit("cuobjdump did not report any CUDA targets for causal_conv1d_cuda")
unexpected = sorted(found - requested)
missing = sorted(requested - found)
if unexpected or missing:
    raise SystemExit(
        f"causal-conv1d CUDA target verification failed: requested={sorted(requested)}, "
        f"found={sorted(found)}, unexpected={unexpected}, missing={missing}"
    )
print(",".join(sorted(found)))
PY
)"
    CTM_CAUSAL_CONV1D_VERIFICATION_STATUS="verified"
    echo "causal-conv1d CUDA targets verified: $CTM_CAUSAL_CONV1D_FOUND_ARCHES"
else
    echo "WARNING: causal-conv1d CUDA target verification disabled by CTM_CAUSAL_CONV1D_VERIFY_ARCHES=0."
fi

"$CTM_VALIDATION_PYTHON" - \
    "$CTM_CAUSAL_CONV1D_PROVENANCE_FILE" \
    "$CTM_CAUSAL_CONV1D_VERSION" \
    "$CTM_CAUSAL_CONV1D_INSTALLED_VERSION" \
    "$CTM_CAUSAL_CONV1D_ARCH_LIST" \
    "$CTM_CAUSAL_CONV1D_ARCHIVE" \
    "$CTM_CAUSAL_CONV1D_ACTUAL_SHA256" \
    "$CTM_CAUSAL_CONV1D_SETUP_SHA256" \
    "$CTM_CAUSAL_CONV1D_PATCHED_SETUP_SHA256" \
    "$CTM_CAUSAL_CONV1D_PATCH_ID" \
    "$CTM_CAUSAL_CONV1D_SOURCE_DIR" \
    "$CTM_CAUSAL_CONV1D_VERIFICATION_STATUS" \
    "$CTM_CAUSAL_CONV1D_FOUND_ARCHES" \
    "$CTM_CAUSAL_CONV1D_FATBIN_FILE" <<'PY'
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

(
    output,
    requested_version,
    installed_version,
    arch_list,
    archive,
    source_sha256,
    upstream_setup_sha256,
    patched_setup_sha256,
    patch_id,
    source_dir,
    verification_status,
    found_arches,
    fatbin_file,
) = sys.argv[1:]

import torch

device = None
if torch.cuda.is_available():
    device = {
        "name": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
    }

record = {
    "schema": "ctm-causal-conv1d-build-provenance-v1",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "package": "causal-conv1d",
    "requested_version": requested_version,
    "installed_version": installed_version,
    "source_archive": archive,
    "source_sha256": source_sha256,
    "upstream_setup_py_sha256": upstream_setup_sha256,
    "patched_setup_py_sha256": patched_setup_sha256,
    "patch_id": patch_id,
    "source_dir": source_dir,
    "target_arch_list": arch_list,
    "force_source_build": True,
    "verification": {
        "status": verification_status,
        "found_arches": found_arches.split(",") if found_arches else [],
        "fatbin_listing": fatbin_file or None,
    },
    "torch": {
        "version": torch.__version__,
        "cuda": torch.version.cuda,
        "device_0": device,
    },
}

target = Path(output)
with target.open("x", encoding="utf-8") as handle:
    json.dump(record, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(target)
PY
echo "causal-conv1d build provenance: $CTM_CAUSAL_CONV1D_PROVENANCE_FILE"

# The upstream vLLM image can contain unrelated optional system-package
# metadata conflicts (for example desktop pygobject). Record them, but let the
# explicit imports and preflight below decide whether this training stack works.
"$CTM_VALIDATION_PYTHON" -m pip check || echo "WARNING: base-image pip metadata has conflicts; explicit validation follows."
"$CTM_VALIDATION_PYTHON" - <<'PY'
from importlib.metadata import version

import causal_conv1d
import fla
import peft
import torch
import transformers
import vllm

print("torch", torch.__version__, "cuda", torch.version.cuda)
for package in ("transformers", "vllm", "peft", "flash-linear-attention", "causal-conv1d"):
    print(package, version(package))
print("validation environment ready", transformers.__version__, vllm.__version__, peft.__version__)
PY
