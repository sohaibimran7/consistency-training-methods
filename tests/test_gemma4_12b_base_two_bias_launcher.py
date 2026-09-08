"""Static custody checks for the Gemma 4 base-evaluation launcher."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace


def _load_launcher():
    path = Path(__file__).parents[1] / "infra/isambard/run_gemma4_12b_base_two_bias_evals_16gpu.py"
    spec = importlib.util.spec_from_file_location("gemma4_12b_launcher_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _linked_snapshot(tmp_path: Path, revision: str) -> tuple[Path, Path]:
    cache = tmp_path / "models--google--gemma-4-12B-it"
    blobs = cache / "blobs"
    snapshot = cache / "snapshots" / revision
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)

    def link(name: str, content: bytes) -> Path:
        digest = hashlib.sha256(content).hexdigest()
        blob = blobs / digest
        blob.write_bytes(content)
        logical = snapshot / name
        logical.symlink_to(Path("../../blobs") / digest)
        return logical

    link("config.json", b'{"model_type":"gemma4_unified"}')
    link("tokenizer.json", b'{"model":"unit-tokenizer"}')
    link("tokenizer_config.json", b'{"tokenizer_class":"Gemma4Tokenizer"}')
    link("processor_config.json", b'{"processor_class":"Gemma4Processor"}')
    link("chat_template.jinja", b"{{ bos_token }}{% for message in messages %}{{ message.content }}{% endfor %}")
    weight = link("model-00001-of-00001.safetensors", b"unit-gemma-weight")
    return snapshot, weight


def test_snapshot_identity_allows_hf_blob_links_and_binds_their_content(tmp_path, monkeypatch):
    module = _load_launcher()
    snapshot, weight = _linked_snapshot(tmp_path, module.MODEL_REVISION)
    monkeypatch.setenv("CTM_GEMMA4_12B_SNAPSHOT", str(snapshot))

    identity = module._snapshot_identity()

    config = identity["assets"]["config.json"]
    assert config["logical_path"] == str(snapshot / "config.json")
    assert config["resolved_path"].startswith(str(snapshot.parent.parent / "blobs"))
    assert len(config["sha256"]) == 64
    weight_record = identity["weight_files"][0]
    assert weight_record["logical_path"] == str(weight)
    assert weight_record["resolved_path"].startswith(str(snapshot.parent.parent / "blobs"))
    assert weight_record["content_address"] == Path(weight_record["resolved_path"]).name
    assert len(weight_record["content_address"]) == 64


def test_discarded_smoke_ids_do_not_overlap_the_scored_first_50():
    module = _load_launcher()
    frozen_ids = tuple(f"question-{index:03d}" for index in range(100))

    smoke_ids = module._discarded_smoke_ids(frozen_ids)

    assert smoke_ids == ("question-050", "question-051")
    assert set(smoke_ids).isdisjoint(frozen_ids[: module.QUESTIONS_PER_CELL])


def test_gpu_wrappers_set_writable_compiler_caches_before_runtime_probe():
    repo = Path(__file__).parents[1]
    for name in (
        "run_gemma4_12b_base_two_bias_smoke.sbatch",
        "run_gemma4_12b_base_two_bias_evals_16gpu.sbatch",
    ):
        script = (repo / "infra" / "isambard" / name).read_text(encoding="utf-8")
        assert script.index("export TORCHINDUCTOR_CACHE_DIR=") < script.index('source "$runtime_env"')
        assert script.index("export XDG_CACHE_HOME=") < script.index('source "$runtime_env"')


def test_deployment_contract_identity_is_stable_across_write_and_resume():
    module = _load_launcher()
    identity = {"schema": "deployment-v1", "manifest_sha256": "a" * 64}

    written = module._stable_deployment_record({**identity, "status": "written"})
    resumed = module._stable_deployment_record({**identity, "status": "resumed"})

    assert written == resumed == identity


def test_worker_command_cannot_pass_a_live_unbiased_log_to_generation(tmp_path, monkeypatch):
    module = _load_launcher()
    snapshot = tmp_path / module.MODEL_REVISION
    snapshot.mkdir()
    monkeypatch.setenv("CTM_GEMMA4_12B_SNAPSHOT", str(snapshot))
    command = module._command_for_worker(
        python="/runtime/python",
        contract={"model_snapshot": {"path": str(snapshot)}},
        paths={"deployment_manifest": tmp_path / "deployment.json"},
        rank=0,
        log_dir=tmp_path / "rank-000",
    )

    task_args = json.loads(command[command.index("--task-args") + 1])
    assert task_args["shard_index"] == 0
    assert "unbiased_log" not in task_args


def test_generation_only_task_strips_exactly_the_standard_live_switch_scorer():
    module = _load_launcher()
    from ctm_data.adapters.mcq_bias.scorer_compat import install_conditional_nan_compat
    from inspect_ai.scorer._scorer import as_scorer_spec
    import mcq_bias.scorers as scorers

    install_conditional_nan_compat()
    sentinel = module._GENERATION_ONLY_SWITCH_SENTINEL
    task = SimpleNamespace(
        scorer=[
            scorers.mcq_bias_scorer(),
            scorers.options_considered_scorer(),
            scorers.switch_scorer(sentinel, question_ids_from=["one"]),
        ],
        task_args={"unbiased_log": sentinel},
        metadata={"unbiased_log": sentinel},
    )

    stripped = module._strip_live_switch_scorer(task)

    assert stripped is task
    assert [as_scorer_spec(item).scorer for item in task.scorer] == [
        "mcq_bias_scorer",
        "options_considered_scorer",
    ]
    assert "unbiased_log" not in task.task_args
    assert "unbiased_log" not in task.metadata


def test_real_inspect_stage2_factory_handles_absent_inner_task_args_across_all_ranks(tmp_path, monkeypatch):
    """Exercise the actual Inspect/``mcq_bias`` task objects, not a fake Task.

    The outer task-factory arguments are not present on the inner Task objects
    built by ``stage2_ood_biased``.  All 16 ranks must therefore be able to
    retain the two generation scorers and remove the construction-only live
    switch dependency without inventing a ``task_args`` field.
    """

    module = _load_launcher()
    from inspect_ai.scorer._scorer import as_scorer_spec

    frozen = tmp_path / "actual-frozen-biased.jsonl"
    ids = tuple(f"question-{index:03d}" for index in range(module.QUESTIONS_PER_CELL))
    rows = [
        {
            "question": f"Question {question_id}",
            "question_id": question_id,
            "source_dataset": "logiqa",
            "prompt_style": "none",
            "unbiased_messages": [{"role": "user", "content": f"Clean {question_id}"}],
            "biased_messages": [{"role": "user", "content": f"Biased {question_id}"}],
            "bias_type": "wrong_argument",
            "ground_truth": "A",
            "biased_option": "B",
            "biasing_text": "A frozen bias signal.",
        }
        for question_id in ids
    ]
    frozen.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    specs = [
        SimpleNamespace(
            kind="unbiased" if task_index <= 3 else "biased",
            frozen_file=str(frozen),
            dataset="logiqa",
            regime="iid",
            population="in_domain",
            bias_type=None if task_index <= 3 else "wrong_argument",
            question_ids=ids,
            source_identity_digest="stage2-ood-hle-2x2:" + "a" * 64,
        )
        for task_index in range(1, module.TASK_COUNT + 1)
    ]
    # This regression is about concrete Task construction and scorer custody,
    # not the GPU-only global EOS patch; the task factory's no-cap parameter
    # remains asserted by the launcher before any execution.
    monkeypatch.setattr(module, "_install_eos_sampler", lambda: None)
    monkeypatch.setattr(module, "_load_specs", lambda _manifest: specs)

    for rank in range(module.SHARD_COUNT):
        tasks = module.gemma_stage2_shard_tasks(manifest=str(tmp_path / "manifest.json"), shard_index=rank)
        assert len(tasks) == module.TASK_COUNT
        for task in tasks:
            assert "unbiased_log" not in task.metadata
            assert getattr(task, "task_args", None) is None
            assert [as_scorer_spec(item).scorer for item in task.scorer] == [
                "mcq_bias_scorer",
                "options_considered_scorer",
            ]


def test_post_merge_standard_switch_score_bypasses_rotated_shard_discovery(tmp_path, monkeypatch):
    """The exact canonical clean path is essential after rotated sharding.

    This uses the real Inspect score API and installed mcq_bias switch scorer,
    rather than a mock scorer.  The two rank-local shards are intentionally
    disjoint, while the merged 50-ID pair is exact and can therefore be scored
    with no directory matching or wait.
    """

    module = _load_launcher()
    source_ids = tuple(f"question-{index:03d}" for index in range(100))
    full_ids = list(source_ids[: module.QUESTIONS_PER_CELL])
    clean_rank_zero = module._shard_ids(source_ids, task_index=1, shard_index=0)
    biased_rank_zero = module._shard_ids(source_ids, task_index=4, shard_index=0)
    assert set(clean_rank_zero).isdisjoint(biased_rank_zero)
    assert len(clean_rank_zero) == 4
    assert len(biased_rank_zero) == 3

    from inspect_ai.log import EvalConfig, EvalDataset, EvalLog, EvalSample, EvalSpec, write_eval_log
    from inspect_ai.model import ModelOutput
    from inspect_ai.scorer import Score
    import mcq_bias.unbiased_log as unbiased_log

    def sample(question_id: str, answer: str, *, biased: bool) -> EvalSample:
        metadata = {
            "variant": "biased" if biased else "unbiased",
            "source_dataset": "logiqa",
            "prompt_style": "none",
            "bias_type": "wrong_argument" if biased else None,
            "biased_option": "B" if biased else "",
        }
        if biased:
            metadata["biasing_text"] = "A frozen bias signal."
        return EvalSample(
            id=question_id,
            epoch=1,
            input="Question\n(A) first\n(B) second",
            target="A",
            output=ModelOutput(
                completion=f"Final answer: {answer}", metadata={"ctm_termination": "model_eos_only"}
            ),
            scores={
                "mcq_bias_scorer": Score(value={"answer_parsed": 1.0}, answer=answer),
                "options_considered_scorer": Score(value={"options_considered": 1.0}),
            },
            metadata=metadata,
        )

    def log(task: str, *, biased: bool) -> EvalLog:
        task_args = {
            "dataset": "logiqa",
            "bias_type": "wrong_argument" if biased else None,
            "question_ids_from": full_ids,
            "prompt_style": "none",
        }
        samples = [
            # First is eligible and switches to B; second already matches B
            # and must be excluded from the conditional towards-rate denominator.
            sample(question_id, "B" if biased or index == 1 else "A", biased=biased)
            for index, question_id in enumerate(full_ids)
        ]
        return EvalLog(
            status="success",
            eval=EvalSpec(
                created="2026-09-08T00:00:00Z",
                task=task,
                task_args=task_args,
                dataset=EvalDataset(name="logiqa", samples=len(full_ids), sample_ids=full_ids),
                model="hf/example",
                config=EvalConfig(limit=len(full_ids)),
            ),
            samples=samples,
        )

    clean_path = tmp_path / "exact-clean.eval"
    write_eval_log(log("stage2_ood_unbiased", biased=False), str(clean_path))
    generation_only_biased = log("stage2_ood_biased", biased=True)

    def unexpected_directory_match(*_args, **_kwargs):
        raise AssertionError("an exact clean EvalLog path must not use directory matching")

    monkeypatch.setattr(unbiased_log, "_matches_unbiased", unexpected_directory_match)
    scored, custody = module._score_merged_biased(
        generation_only_biased,
        clean_path=clean_path,
        full_ids=full_ids,
    )

    # ``copy=True`` preserves the generation-only merged object for hashing;
    # the canonical scored output is a separate derived object.
    assert scored is not generation_only_biased
    assert "switch_scorer" not in generation_only_biased.samples[0].scores
    assert custody["mode"] == "post_merge_cpu_standard_inspect"
    assert custody["scorer"] == "mcq_bias/switch_scorer"
    assert custody["rescore_model"] == "mockllm/model"
    assert custody["clean_log"]["path"] == str(clean_path.resolve())

    eligible = scored.samples[0].scores["switch_scorer"]
    already_biased = scored.samples[1].scores["switch_scorer"]
    assert eligible.value["towards_bias_switch"] == 1.0
    assert math.isnan(eligible.value["away_from_bias_switch"])
    assert math.isnan(already_biased.value["towards_bias_switch"])
    assert already_biased.value["away_from_bias_switch"] == 0.0
    for scored_sample in scored.samples:
        score = scored_sample.scores["switch_scorer"]
        assert score.metadata["unbiased_log"] == str(clean_path.resolve())
        assert "note" not in score.metadata

    # The CPU-scored canonical output is persisted before publication.  Its
    # NaN conditional-exclusion sentinel must survive the real EvalLog
    # round-trip and be accepted by the consumer as an excluded denominator
    # row, rather than a malformed switch score.
    from experiments.gemma4_12b_base_eval import postprocess
    from inspect_ai.log import read_eval_log

    scored_path = tmp_path / "canonical-scored-biased.eval"
    write_eval_log(scored, str(scored_path))
    persisted = read_eval_log(str(scored_path), header_only=False)
    postprocess._validate_source_samples(persisted, kind="biased", bias_type="wrong_argument")
    assert postprocess._switch_value(persisted.samples[1]) is None

    # Build a complete receipt-bound 21-cell campaign around this actual
    # 50-question standard-scored log.  This exercises the publication
    # consumer rather than merely its single-sample helper: all sources are
    # serialized canonical EvalLogs, the one real score retains its NaN
    # conditional exclusion, and preflight must accept the full matrix.
    campaign = tmp_path / "campaign"
    merged_root = campaign / "merged"
    shard_records = []
    for rank in range(module.SHARD_COUNT):
        shard = campaign / "live-shards" / f"rank-{rank:03d}" / "immutable-generation.eval"
        shard.parent.mkdir(parents=True, exist_ok=True)
        shard.write_text(f"immutable-rank-{rank}\n", encoding="utf-8")
        shard_records.append(
            {
                "rank": rank,
                "raw_log": {
                    "path": str(shard.resolve()),
                    "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
                    "size_bytes": shard.stat().st_size,
                },
            }
        )

    (campaign / "launch-contract.json").write_text(
        json.dumps(
            {
                "schema": postprocess.LAUNCH_SCHEMA,
                "campaign": postprocess.CAMPAIGN_NAME,
                "model_snapshot": {"model_id": postprocess.MODEL_ID, "revision": postprocess.MODEL_REVISION},
                "sampling": postprocess.GENERATION_SAMPLING,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    clean_template = log("stage2_ood_unbiased", biased=False)

    def clone_cell(template, *, dataset: str, bias_type: str | None):
        task_args = dict(template.eval.task_args)
        task_args.update({"dataset": dataset, "bias_type": bias_type})
        dataset_record = module._copy_update(template.eval.dataset, name=dataset)
        evaluation = module._copy_update(
            template.eval,
            task="stage2_ood_biased" if bias_type is not None else "stage2_ood_unbiased",
            task_args=task_args,
            dataset=dataset_record,
        )
        samples = []
        for source_sample in template.samples:
            metadata = dict(source_sample.metadata)
            metadata.update(
                {
                    "variant": "biased" if bias_type is not None else "unbiased",
                    "source_dataset": dataset,
                    "bias_type": bias_type,
                }
            )
            if bias_type is None:
                metadata.pop("biasing_text", None)
            else:
                metadata["biasing_text"] = "A frozen bias signal."
            samples.append(module._copy_update(source_sample, metadata=metadata))
        return module._copy_update(template, eval=evaluation, samples=samples, results=None)

    cells = [
        (dataset, bias_type)
        for dataset in postprocess.DATASETS
        for bias_type in (None, *postprocess.ALL_BIASES)
    ]
    assert len(cells) == module.TASK_COUNT
    for task_index, (dataset, bias_type) in enumerate(cells, start=1):
        template = scored if bias_type is not None else clean_template
        canonical_log = clone_cell(template, dataset=dataset, bias_type=bias_type)
        task_root = merged_root / f"task-{task_index:03d}"
        task_root.mkdir(parents=True)
        staged = task_root / "staged.eval"
        write_eval_log(canonical_log, str(staged))
        raw_identity = {
            "path": str(staged.resolve()),
            "sha256": hashlib.sha256(staged.read_bytes()).hexdigest(),
            "size_bytes": staged.stat().st_size,
        }
        canonical = task_root / f"{raw_identity['sha256']}.eval"
        staged.replace(canonical)
        raw_identity["path"] = str(canonical.resolve())
        full_digest = hashlib.sha256(
            json.dumps(full_ids, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        receipt = {
            "schema": postprocess.MERGE_RECEIPT_SCHEMA,
            "campaign": postprocess.CAMPAIGN_NAME,
            "task_index": task_index,
            "kind": "biased" if bias_type is not None else "unbiased",
            "regime": "ood" if dataset == "hle-text-mc" else "iid",
            "population": "held_out" if dataset == "hle-text-mc" else "held_in",
            "dataset": dataset,
            "bias_type": bias_type,
            "sample_count": module.QUESTIONS_PER_CELL,
            "full_question_ids_sha256": full_digest,
            "shard_count": module.SHARD_COUNT,
            "shard_logs": shard_records,
            "raw_log": raw_identity,
        }
        receipt_path = merged_root / "receipts" / f"task-{task_index:03d}.json"
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")

    preflight = postprocess.preflight_campaign(campaign_root=campaign)
    assert preflight["cells"] == 21
    assert preflight["clean_cells"] == 3
    assert preflight["biased_cells"] == 18
