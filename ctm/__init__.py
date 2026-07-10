"""ctm — consistency-training methods, backend-agnostic.

Package layout (grown strangler-fig style; old import paths in
``cot_transparency.apis.tinker`` re-export from here):

- ``ctm.core``      — pure math and types: rewards, advantage/SNR estimators, configs.
- ``ctm.backends``  — the compute seam: Tinker service adapter + local (torch/PEFT) engine,
                      behind one protocol (sample / forward_backward / optim_step / checkpoints).
- ``ctm.training``  — the RL (RLCT) and SFT (BCT) loops, backend-injected.
- ``ctm.settings``  — pluggable phenomena (sycophancy, eval-awareness, ...); phase 2.
- ``ctm.evals``     — setting-agnostic eval machinery and analysis.
"""
