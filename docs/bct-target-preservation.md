# Complete-response BCT targets

BCT targets must retain the frozen clean model's entire generated assistant
response, including reasoning and the final answer. The evaluation helper
`decode_response` is not a BCT target serializer: its structured-content path
intentionally extracts display text only.

`ctm.training.bct_response` is the separate target path. Cookbook messages retain
ThinkingPart/TextPart content. Local HF targets bypass the final-only response
parser. GPT-OSS uses its native `thinking` field; Gemma's unified processor uses
its native `reasoning` field; Qwen's generated think content retains its framing.
Both paired prompts are rendered and checked before a
target is accepted: substantive reasoning and final text must survive in tokens
with positive loss weight. Unknown content structures fail closed. The exact
Qwen opening think prefill can be prompt-masked, but reasoning cannot.

Progress schema 2 and the explicit target policy prevent silently reusing old
generation progress. New generated rows record the policy and sampled-token
hash. Existing immutable datasets are not rewritten, and this change cannot
recover reasoning previously omitted from a saved dataset.

Tests in `tests/test_bct_response_preservation.py` exercise actual cookbook
renderers with a synthetic lossless tokenizer, including the supervised mask.
They are not substitutes for native cached model tokenizer/template checks.
Model-specific CPU receipts must accompany compatibility claims. No model
inference is needed for those synthetic response checks.

Run `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/check_bct_preservation.py
/absolute/cached/snapshot qwen` (or `gemma` / `gptoss`) in that model's supported
runtime. The probe loads configuration, tokenizer and processor only, prints a
JSON receipt including asset/helper hashes and token weights, and exits nonzero
on failure. It must not be interpreted as a historical training audit.

## Historical scope

The shared CTM target-generation path is distinct from the two-bias convergence
trainer's raw-token cache and direct token-weight construction. Neither this
fix nor a current synthetic renderer test proves that a historical run lost
reasoning. Audit the saved targets, actual consumer, and runtime/source binding
per run. Original cot-transparency/Tinker GPT-OSS runs require their original
code and dataset provenance, not a CTM introduction date.
