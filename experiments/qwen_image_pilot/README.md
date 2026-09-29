This is an authorized feasibility pilot of Qwen3.5-9B and the RMCT step-352
adapter using image inputs. It uses image renderer source commit
`e6a616b480760861950c78c8b46429af00ea5f23` from
https://github.com/laurenceburnsmill/mcq-bias-image.

The 93 cases per model use three frozen HLE questions: the two longest and the
median by clean prompt length from the existing 100-question evaluation pool.
They cover the four available text overlays, four colours each for ticks and
circles, and separate/composite spurious and wrong-few-shot demonstrations.
Matched controls include no artifact, no few shots, and clean targets with
the artifact still present on correct demonstration answers. Tick/circle
controls use identical renderer layouts without marks.

Spurious examples reuse the upstream five-example corpus and seeded shuffle.
Wrong-few-shot demonstrations are extracted from the existing frozen HLE
wrong-few-shot prompts, retaining their correct labels and ordering. The
target is not included among demonstrations: its wrong label is applied only
to the target image, as explicitly requested for this image pilot.

The upstream ordinary image renderer appends `biasing_text`; it does not
reproduce post-hoc and are-you-sure conversational turns. Those cases retain
and label the upstream semantics. No wrong-argument store is exposed by the
image CLI, so that condition is excluded.

The output cap (20,480), temperature (1), top-p (.95), and top-k (20) reuse the
approved RMCT evaluation configuration. Thinking is enabled. Both conditions
use identical image bytes and seed 42. Serving uses the full multimodal
Qwen wrapper, 65,536 context, and the existing translated RMCT adapter.
The existing raw checkpoint and translated checkpoint are checked tensor by
tensor. This tensor audit alone does not establish runtime adapter activation.

`audit` reconstructs PNGs from the exact processor pixel tensors after
normalization reversal, records image grids and visual token counts, and
checks context headroom. Inspect logs embed the input images, full responses,
API records, scores, and image/manifest hashes. The strict answer parser runs
on the final answer outside explicitly delimited reasoning; unparsed responses
are reported separately. The small, length-selected sample supports interface
and artifact checks, not population estimates or significance tests.

Remote run directory:
`/projects/a5v/sohaib.a5v/ctm-qwen-image-pilot-20260920`.

Initial job `6722394` completed zero samples: FlashInfer DeltaNet needed nvcc.
Retry `6722484` exposed a FlashInfer/CUDA compilation incompatibility before
inference. Its stuck server was stopped; no other jobs were touched.
Recovery `6722531` selects the supported `gdn_prefill_backend=triton`, with one
GPU and a three-hour walltime. Inspect explicitly uses Chat Completions because
the Responses route discarded the sampling seed in the initial attempt.

Processor audit: the longest target changes from 675x5430 to 672x5440, and
the longest wrong-few-shot composite changes from 644x8142 to 640x8128.
There is no crop; dimensions round to patch-grid multiples. The largest case
uses 7,680 visual tokens. Processed PNGs reconstructed from pixel tensors were
visually inspected for readable text and visible geometric square markers.
