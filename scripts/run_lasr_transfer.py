"""Run only the fixed, archive-verified LASR budget pilot on attested local ports.

Audit never contacts a model endpoint. Smoke/pilot perform generation only;
scorers are absent and Inspect scoring is explicitly disabled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lasr_transfer_tasks import MODELS, TARGET_CONFIG, audit_summary, build_pilot_tasks

PORTS = {"base": 18000, "mo_mid": 18001, "mo_post": 18002}


def validate_attestation(attestation: dict, role: str) -> None:
    """Reject old text-only server attestations and mismatched runtime flags."""
    expected_id, expected_revision = MODELS[role]
    if (attestation.get("model"), attestation.get("revision")) != (expected_id, expected_revision):
        raise ValueError("LASR endpoint model pin mismatch")
    expected = {
        "role": role,
        "reasoning_parser": "qwen3",
        "enable_auto_tool_choice": True,
        "tool_call_parser": "hermes",
        "flash_attention_version": 3,
        "authenticated_api": True,
        "sampler": "native",
    }
    if any(attestation.get(key) != value for key, value in expected.items()):
        raise ValueError("LASR requires a fresh Hermes/Qwen3/FA3 runtime attestation")
    runtime = attestation.get("runtime", {})
    if runtime.get("vllm") != "0.19.1" or len(runtime.get("gpus", [])) != 4:
        raise ValueError("LASR target requires vLLM 0.19.1 on four GPUs")
    command = attestation.get("launch_command", [])
    if not isinstance(command, list) or "serve" not in command or "--enable-auto-tool-choice" not in command:
        raise ValueError("Missing attested native tool-serving command")
    flags = {
        "--served-model-name": expected_id,
        "--reasoning-parser": "qwen3",
        "--tool-call-parser": "hermes",
        "--tensor-parallel-size": "4",
        "--max-model-len": "32768",
        "--dtype": "bfloat16",
        "--attention-backend": "FLASH_ATTN",
    }
    for flag, value in flags.items():
        if command.count(flag) != 1 or command[command.index(flag) + 1] != value:
            raise ValueError("LASR attested server flags do not match the fixed protocol")
    attention_index = command.index("--attention-config")
    if json.loads(command[attention_index + 1]) != {"flash_attn_version": 3}:
        raise ValueError("LASR attention configuration mismatch")
    snapshot = command[command.index("serve") + 1]
    if Path(snapshot).name != expected_revision:
        raise ValueError("LASR attested snapshot is not revision-pinned")


def resolve_live_runtime(role: str, runtime_dir: Path, *, opener=None):
    """Read a dedicated key internally and check the authenticated served model."""
    opener = opener or urllib.request.urlopen
    attestation_path = runtime_dir / f"{role}-attestation.json"
    payload = attestation_path.read_bytes()
    attestation = json.loads(payload)
    validate_attestation(attestation, role)
    key = (runtime_dir / "api-key").read_text().strip()
    if not key:
        raise ValueError("Missing dedicated LASR endpoint credential")
    endpoint = f"http://127.0.0.1:{PORTS[role]}/v1"
    request = urllib.request.Request(endpoint + "/models", headers={"Authorization": "Bearer " + key})
    with opener(request, timeout=20) as response:
        served = json.load(response).get("data", [])
    if len(served) != 1 or served[0].get("id") != MODELS[role][0]:
        raise ValueError("Live LASR served model does not match the role")
    if served[0].get("max_model_len", 32768) != 32768:
        raise ValueError("Live LASR context length mismatch")
    proof = {
        "endpoint": endpoint,
        "runtime_attestation_sha256": hashlib.sha256(payload).hexdigest(),
        "model_id": MODELS[role][0],
        "model_revision": MODELS[role][1],
        "vllm": "0.19.1",
        "tensor_parallel_size": 4,
        "flash_attention_version": 3,
        "reasoning_parser": "qwen3",
        "tool_call_parser": "hermes",
        "enable_auto_tool_choice": True,
        "authenticated_live_models_check": True,
    }
    return endpoint, key, proof


def run(args) -> dict:
    log_dir = Path(args.log_dir)
    if log_dir.exists():
        raise ValueError("Log directory already exists; select a fresh path")
    tasks = build_pilot_tasks(source_dir=args.source_dir, archive=args.archive, model_role=args.role, stage=args.stage)
    summary = audit_summary(tasks, args.stage)
    summary["model_role"] = args.role
    summary["log_dir"] = str(log_dir)
    model = None
    if args.stage != "audit":
        if not args.runtime_dir:
            raise ValueError("Live runs require --runtime-dir")
        endpoint, key, proof = resolve_live_runtime(args.role, Path(args.runtime_dir))
        summary["runtime"] = proof
        from inspect_ai.model import GenerateConfig, get_model

        model = get_model(
            "vllm/" + MODELS[args.role][0],
            base_url=endpoint,
            api_key=key,
            config=GenerateConfig(**TARGET_CONFIG),
        )
    log_dir.mkdir(parents=True, exist_ok=False)
    with (log_dir / "run-manifest.json").open("x") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    printable = {key: value for key, value in summary.items() if key not in {"cells", "prompt_audits"}}
    print(json.dumps(printable, indent=2), flush=True)
    if args.stage == "audit":
        return printable
    import inspect_ai

    logs = inspect_ai.eval(
        tasks=tasks,
        model=model,
        epochs=1,
        score=False,
        log_dir=str(log_dir),
        metadata={"lasr_transfer_run": printable},
        max_tasks=4,
        display="plain",
    )
    if any(log.status != "success" for log in logs):
        raise RuntimeError("One or more LASR generation tasks failed")
    result = {"status": "generation_complete", "role": args.role, "stage": args.stage, "episodes": len(tasks)}
    print(json.dumps(result), flush=True)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--role", choices=MODELS, required=True)
    parser.add_argument("--runtime-dir", help="Dedicated LASR credential and fresh role-attestation directory")
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--stage", choices=("audit", "smoke", "pilot"), required=True)
    args = parser.parse_args(argv)
    try:
        run(args)
    except Exception as error:
        # Provider exceptions can contain private traces or request details.
        print(json.dumps({"status": "failed", "error_type": type(error).__name__}), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
