"""Frozen identities and ordering for the RMCT TBSR evaluation."""

from __future__ import annotations

MODELS = (
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3-8B",
)

TRAINING_DATASETS = ("logiqa", "hellaswag")
AUTHORITATIVE_TRAINING_COUNTS = {"logiqa": 472, "hellaswag": 1576}
AUTHORITATIVE_TRAINING_ROWS = 2048
TRAINING_COUNTS = {"logiqa": 100, "hellaswag": 100}
TRAINING_ROWS = 200
TRAINING_SELECTION_SEED = 20260729
TRAINING_SPLIT = "rmct-training-balanced-n200-seed20260729"
SOURCE_ROWS = 2390
SOURCE_SHA256 = "d6eabbfe0ad5c5401702b256c69f7fd9c8c19be1b745935cd86676513092ffab"
TRAINING_SHA256 = "517b89de2c1dd724c3250e9c2b7847ac1665e469d6b61e3368943be09573d7fc"

HLE_DATASET = "hle-text-mc"
HLE_SPLIT = "hle"
HLE_ROWS = 100
HLE_BIASES = (
    "suggested_answer",
    "distractor_fact",
    "wrong_argument",
    "post_hoc",
    "spurious_few_shot_squares",
    "wrong_few_shot",
)
HLE_PAPER_HELD_OUT_BIASES = tuple(bias for bias in HLE_BIASES if bias != "wrong_argument")

DEFAULT_SOURCE = "artifacts/switch-gate/source/distractor-argument-pairs.jsonl"
DEFAULT_TRAINING_OUTPUT = "artifacts/rmct-tbsr/training-balanced-n200-seed20260729.jsonl"
DEFAULT_TRAINING_MANIFEST = "artifacts/rmct-tbsr/training-balanced-n200-seed20260729.manifest.json"
DEFAULT_HLE_DIR = "artifacts/switch-gate/source/hle-eval"

HLE_FILES = {
    "unbiased": "hle-text-mc_unbiased_none_n100_seed42_ids-1dc073edc4.jsonl",
    "suggested_answer": "hle-text-mc_suggested_answer_none_n100_seed42_ids-1dc073edc4.jsonl",
    "distractor_fact": "hle-text-mc_distractor_fact_none_n100_seed42_ids-1dc073edc4.jsonl",
    "wrong_argument": (
        "hle-text-mc_wrong_argument_none_n100_seed42_args-vllm-google-gemma-4-31b-it_ids-1dc073edc4.jsonl"
    ),
    "post_hoc": "hle-text-mc_post_hoc_none_n100_seed42_ids-1dc073edc4.jsonl",
    "spurious_few_shot_squares": ("hle-text-mc_spurious_few_shot_squares_none_n100_seed42_ids-1dc073edc4.jsonl"),
    "wrong_few_shot": "hle-text-mc_wrong_few_shot_none_n100_seed42_ids-1dc073edc4.jsonl",
}

HLE_FILE_SHA256 = {
    "unbiased": "542b22b1531fa48d5aeb5a1716db100d41430a7e5b95079794300837364839e1",
    "suggested_answer": "c7e42217682469029657340ebebf6a093c711390a8927b635e7a76db51de4617",
    "distractor_fact": "1883d27c55502b6e188c6747f6c1c795d7cb6c7207fd577fdfc57329fd1ec076",
    "wrong_argument": "8a91aca8ef2b9286b83d93605e654e1135fbcbbbb45a6d5e6ba2675ed7abf2bf",
    "post_hoc": "b02ae2c1d02a4ac3e98d12906a3e4053f4f7ecd53361bc2379bb5bcdc8066eb2",
    "spurious_few_shot_squares": "400faf1e3fc60bcb47253a017d8572dfc5d6b00369308681e1f85ae16b6f4d57",
    "wrong_few_shot": "ea2317dae1a4dfe22cefd462e8fef8ee41c1dacdabb35727a7e197e98e291cd2",
}
