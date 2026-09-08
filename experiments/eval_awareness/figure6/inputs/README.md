# EvalAwareBench Figure 6 inputs

These are immutable inputs for the Qwen-only Figure 6 reproduction. Generated
responses, judgments, plots, and Inspect logs belong under `artifacts/` and are
not tracked by Git.

| File | Purpose | SHA-256 |
|---|---|---|
| `prompts.jsonl` | Materialized 100-task x 2-valence x 9-condition benchmark slice | `eae540adca34bdab00ad0fab200fa6d615e69fe87e7abfa8431324a59e1ac47a` |
| `prompts.jsonl.manifest.json` | Dataset revision, source-file hashes, license, and selection provenance | `4fc4af708e32f206a699728a7e78debb949078ae6221cb45dceab9347c08c81e` |
| `chat_prompt_realistic.txt` | Paper-natural Qwen system prompt | `8d46fa8eee79ba7372088d0bd138a463cc5255c8043a77f468bed937f609d735` |
| `judgePromptEvidenceBoth.txt` | Evaluation-awareness and task-performance judge prompt | `e6158c9dba2466519450f4234e5dc0f9b4c97717b759ba6a133e2233f6dc3870` |

`prompts.jsonl` was deterministically materialized from
`aisa-group/EvalAwareBench` at revision
`736bda484525041972fa214eaf72f3beba492666` using the selection code pinned in
this repository. The source dataset is recorded as CC-BY-NC-4.0 in the adjacent
manifest. The two prompt files come from
`aisa-group/decomposing-eval-awareness` at revision
`446be5c605b56a60d4efe2526f0cbf55522c523a`.

The deferred Llama runs require `chat_prompt_realistic_scratchpad.txt`, which is
not part of the completed Qwen-only experiment and is therefore not included
here. Its upstream filename and pinned digest remain in `protocol.yaml`.
