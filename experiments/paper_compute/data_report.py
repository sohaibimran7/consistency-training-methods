"""Render audit tables and an exact checkpoint path catalogue."""
import json,hashlib
from pathlib import Path
from common import artifact_directory
ROOT=artifact_directory("data-matched-checkpoints-20260926")
a=json.loads((ROOT/'analysis.json').read_text())
v=json.loads((ROOT/'selected-checkpoint-verification.json').read_text())
g=json.loads((ROOT/'gemma-rmct.json').read_text())
METHODS=a['method_order'];names={'bct':'BCT','act':'ACT','attct':'AttCT','mlpct':'MLPCT','opct':'OPCT','rmct':'RMCT'}
for w in g['windows'].values():
 for cp in v:
  if cp['path']==w['checkpoint']:
   assert all(w['checkpoint_hashes'][n]==h for n,h in cp['hashes'].items())
catalogue=['# Exact retained checkpoint paths','', 'These are raw training adapters. Evaluation-specific conversion/activity checks remain required; this audit launches no evaluation.','']
for key,title in [('latest','Latest retained'),('within_family_question_count_candidates','Largest within-family question-count budgets'),('cross_family_question_count_candidates','Shared 576-question budget')]:
 catalogue+=['## '+title,'']
 for fam,ms in a[key].items():
  catalogue+=['### '+fam.title(),'']
  for m,c in ms.items():
   catalogue += [f"- {names[m]}: optimizer update **{c['optimizer_updates']}**, attempted group **{c['attempted_groups']}**, {c['unique_questions_encountered']} distinct questions: `{c['path']}`"]
  catalogue+=['']
catalogue+=['## Exact-QID-sequence non-RMCT subset','', 'Qwen: the five non-RMCT paths at update288 above. Historical Gemma: the following five at update302 (604 QIDs), immediately before OPCT first skips a group.','']
for cp in v:
 if cp['family']=='gemma' and cp['optimizer_updates']==302:catalogue += [f"- {names[cp['method']]}: `{cp['path']}`"]
(ROOT/'CHECKPOINTS.md').write_text('\n'.join(catalogue)+'\n')
lines=['# Data-matched checkpoint audit — 26 September 2026','',
'## Conclusion and recommendation','',
'There is **no positive, exact all-six-method training-QID match** in either family’s retained checkpoints. RMCT’s 1,000-QID pool/order differs from the five other methods’ 7,680-QID pool/order. Equal optimizer steps do not repair that difference. Gemma also has skipped groups and historical thinking-disabled training. No corrected thinking-enabled Gemma checkpoint roster exists according to its owner’s fresh 26 September audit.',
'',
'For an existing-checkpoint **question-count-controlled sensitivity comparison**, use **576 distinct questions encountered** for every model/method: Qwen update288 for all six; historical Gemma update288 for BCT/ACT/AttCT/MLPCT/OPCT and **RMCT optimizer177 / global288**. These exact adapters are retained and file-verified. Label this “matched number of training questions encountered,” **not “identical training data,” “equal gradient exposure,” or a matched thinking-enabled cross-model replication**. It is not a new validation-optimal checkpoint selection.',
'',
'For a genuinely identical-QID-and-order subset, the five non-RMCT methods can use Qwen update288 (576 QIDs) and historical Gemma update302 (604 QIDs). Across both families, those five can use update288 (576 identical QIDs); thinking and method-specific prompt/rollout differences remain. Keep RMCT explicitly separate from any exact-data-matched claim. A strict all-six matched training comparison cannot be recovered by checkpoint selection alone; this report does not authorize retraining.',
'',
'## Latest retained checkpoints','',
'All steps below are actual optimizer updates, not filename/global counters. “Updating-group QIDs” counts question opportunities in groups that led to an update; it does not establish that every question/sample had a nonzero gradient.','',
'“Encountered” means the questions traversed by the saved training-data cursor on the committed lineage, including skipped groups. It excludes abandoned/replayed attempts and speculative prefetch beyond that cursor; it is not an all-sampling or token-cost total.','',
'| Method | Qwen update | Qwen distinct QIDs encountered | Gemma update | Gemma distinct QIDs encountered | Gemma updating-group QIDs |',
'|---|---:|---:|---:|---:|---:|']
for m in METHODS:
 q=a['latest']['qwen'][m];gcp=a['latest']['gemma'][m]
 lines += [f"| {names[m]} | {q['optimizer_updates']} | {q['unique_questions_encountered']} | {gcp['optimizer_updates']} | {gcp['unique_questions_encountered']} | {gcp['questions_in_updating_groups']} |"]
lines += ['', 'Gemma RMCT192 consumed 310 attempted groups: 118 groups did not update, giving 620 encountered QIDs but 384 QID opportunities in updating groups. Gemma OPCT960 consumed 962 groups, excluding attempts302 and558: 1,924 encountered vs1,920 updating-group QIDs. No question repeats occur in these committed trajectories; repeated rollouts and abandoned/replayed work are separate exposure/compute categories.', '',
'## Largest count-only budgets available on disk','',
'| Scope | Distinct QIDs encountered per method | BCT | ACT | AttCT | MLPCT | OPCT | RMCT |',
'|---|---:|---:|---:|---:|---:|---:|---|',
'| Qwen only | 576 | 288 | 288 | 288 | 288 | 288 | optimizer288 / global288 |',
'| Historical Gemma only | 620 | 310 | 310 | 310 | 310 | 309 | optimizer192 / global310 |',
'| Gemma at shared cross-family budget | 576 | 288 | 288 | 288 | 288 | 288 | optimizer177 / global288 |',
'',
'Qwen is limited by BCT’s retained terminal288. Gemma is limited by RMCT’s terminal global310. Gemma OPCT309 has already skipped one group, so using OPCT310 would exceed the620-question budget. The shared budget576 is attainable without a retention-grid approximation.',
'',
'At620 encountered questions, Gemma BCT/ACT/AttCT/MLPCT have620 updating-group QID opportunities, OPCT618 and RMCT384. At576, the five non-RMCT methods have576; Gemma RMCT has354. Consequently these are **not matched optimized-example counts**. Matching optimizer192 instead would give384 nominal updating-group QIDs each, but RMCT would still have encountered620 questions versus384 for the other methods, and individual gradient/sample usability would still differ.',
'',
'## Actual question identity, split and order','',
'- RMCT: 1,000 QIDs, 500 LogiQA +500 HellaSwag; pool SHA256 `cc7842566093e10d3867514e40b55d62ba56521fee24e2ba3695036295199075`; manifest SHA256 `eac0682fe0286126cc5928253e2e2ef4968eee50775a65feb21d061eec6853cc`.',
'- Other methods: 7,680 QIDs, 3,840 per dataset; pool SHA256 `3084f27837a16f5175f1a15066c3e0fda7dd22b4aab16f2dd8d0381c45cee8f7`; manifest SHA256 `e50396d8fb2188f5959f9ced378a8f813922b6015f1a1a73887cfbccc52b43d1`.',
'- Both are frozen wrong-argument-store-derived training populations, with wrong-argument and suggested-answer views of each question. The manifests record protected IID and Stage2 QID exclusions. Rows do not supply an official source train/test split field; do not invent one or equate these training pools with the held-out validation pool. The separate deployed-provenance-20260925 audit records zero overlap with its revised200-QID validation set; this audit does not change that set.',
'- Same ordering algorithm/seed string (`rmct-shared-qid-two-bias-v1`) does not imply same order when the pool changes. Both interleave one LogiQA and one HellaSwag question; no training shuffle. Their entire pools overlap on971 QIDs, but the selected576-question prefixes overlap on only39; the620-question prefixes overlap on46. All eligible retained-budget prefixes were checked: none gives an identical all-method question set.',
'- The five non-RMCT methods share the exact question order. Sealed per-step QIDs were checked against the expanded manifest for every retained committed update. Gemma OPCT first diverges in optimized QID sequence after update302, so604 is its largest exact shared QID-sequence budget with the four other non-RMCT Gemma methods.',
'',
'## What is and is not matched','',
'| Property | Five non-RMCT methods | RMCT |',
'|---|---|---|',
'| Logical question group | 2 QIDs; both biases; 4 paired objectives | 2 QIDs; both biases; reference and biased rollouts |',
'| Physical accumulation | 1 pair/backward, 4 accumulations/update | Batch2, accumulation1; forward microbatching is an execution detail |',
'| Sampling multiplicity | Internal methods: fixed paired views. BCT: one clean target/QID reused for both biases. OPCT: 4 samples/biased prompt (16/update) | 96 clean reference samples/QID and96/biased view; nominal576 completions/attempt before retry/validity handling |',
'| Ordering | Expanded-pool interleave; seed42+zero-based optimizer update for internal training RNG | Small-pool interleave; seed42, rollout seed42; restart/segment RNG handling differs |',
'| LR/Adam | LR1e-4 constant; betas0.9/0.95, eps1e-8, no weight decay, clip1 | Same |',
'| LoRA | rank8, alpha16, dropout0; method-specific attention vs full scopes | rank8, alpha16, dropout0; all text attention+MLP |',
'',
'ACT/AttCT/MLPCT use the approved suggested-answer prefix transformation; BCT/OPCT/RMCT use native cues. Qwen ACT includes fused DeltaNet QKV (includingK); Gemma attention projections are separate. MLPCT measures an MLP objective but trains Q/V attention adapters. These are method/architecture differences, not equal text/token exposure. No Alpaca. Historical Gemma training is thinking-off; Qwen thinking-enabled. Historical Qwen online runs use20,480 generated tokens with retained-length behavior; Gemma’s20,000-token amendment excludes truncations and follows initially uncapped RMCT work. Token/rollout exposure and stochastic sampling are **not** matched by this question budget.',
'',
'## Retention, loadability and limits','',
'- All six methods have retained historical adapters in each family. Qwen retains16-update boundaries through each reported terminal; BCT/OPCT span the preserved cap-v4 and recovery-v5 roots. Non-window Qwen recovery snapshots were rotated; they are not silently substituted. Qwen RMCT has22 committed16-update segments; copied parents and failed attempts are excluded.',
'- Historical Gemma non-RMCT retains every actual update. Gemma RMCT was inspected through244 snapshots (largely post-update45, plus earlier window16/32 endpoints); this is not an assertion that every earlier off-window snapshot survives. Candidate steps and clocks are read from manifests. `_step288` in the selected Gemma RMCT filename means global288, **not optimizer288**.',
'- 33 unique latest/candidate adapters were read and SHA256-hashed on Isambard. Every selected safetensors header had contiguous tensor offsets matching the complete payload size, and model/config/optimizer-clock metadata parsed. 29 non-RMCT adapters matched original per-step receipt hashes; Gemma RMCT terminal192 also matched its window receipt hashes. Fresh hashes for other RMCT candidates are retained in the verification artifact.',
'- This establishes retained structurally readable adapter payloads, **not a new HF/vLLM model-load or numerical parity test**. Usual method-specific adapter conversion and runtime activity checks are still required before evaluation. No GPU model load, inference, training, submission, cancellation, controller update or convergence-state edit was performed.',
'- Do not confuse the historical own-loss terminal/best checkpoints with the source task’s ongoing TBSR validation selection. This audit chooses only by exposure budget and leaves that work intact.',
'',
'## Artifacts and reproduction','',
'- [Exact checkpoint path catalogue](CHECKPOINTS.md).',
'- [Machine-readable comparison and retained-step inventory](analysis.json).',
'- [Selected checkpoint file hashes and structural verification](selected-checkpoint-verification.json).',
'- `remote-inventory.json`: remote non-RMCT manifests, sealed metrics, Qwen RMCT configs/receipts (large raw snapshot).',
'- `gemma-rmct.json`: retained Gemma RMCT manifests, complete-window receipts and optimizer-clock records.',
'- `exposure-receipts.json`: early Gemma windows, Qwen RMCT committed metrics, frozen Gemma source, skipped groups.',
'- Reproduce locally without remote access: `python3 experiments/paper_compute/data_analyze.py --artifact-root /path/to/artifacts` then `python3 experiments/paper_compute/data_report.py --artifact-root /path/to/artifacts`. The collectors are read-only remote scripts; rerunning them refreshes only this audit directory.',
'',
f"Remote inventory snapshot began {a['observed_utc']}. Gemma owner coordination independently confirmed the current historical roster and absence of a corrected thinking-enabled lineage on26 September. All artifacts are isolated from convergence/configuration owners."]
(ROOT/'README.md').write_text('\n'.join(lines)+'\n')
files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.iterdir()) if p.is_file() and p.name!='SHA256.json'}
(ROOT/'SHA256.json').write_text(json.dumps(files,indent=2)+'\n')
print(json.dumps({'report':str(ROOT/'README.md'),'catalogue':str(ROOT/'CHECKPOINTS.md'),'hashed_artifacts':len(files),'gemma_rmct_terminal_receipt_verified':True}))
