# Reproducibility and release checks

Release preparation date: 2026-10-15. Public model scope: the final Full ARGScope-AA and ARGScope-NT checkpoints only.

## Checks actually performed

- Original source, checkpoint and Figure 1 copies are identified with SHA-256 hashes. Exact-copy comparisons and archive CRC checks were performed during packaging.
- The canonical Full/ablation receipts, saved split identities and recorded final student outputs were inspected. Saved prediction labels and independently recalculated metrics were compared by a separate read-only review.
- All included Python source files were parsed for syntax during packaging; this does not execute them or establish that their historical dependencies are still available.
- The root `predict.py` wrapper was tested using both original Full checkpoints and their original 4,574-row test embeddings. Binary, type and subtype predictions and both reported probability columns matched exactly. The detailed, machine-readable check is `INFERENCE_REPLAY.json`.
- Text files selected for redistribution were screened for common credential patterns. Potential matches were quarantined instead of printed or uploaded. This is a bounded automated screen, not a security guarantee.

## Checks not performed

No model was retrained, no parameter was newly selected, no biological sequence was synthesized, and no new experimental validation was performed during release preparation. Historical code is preserved, not represented as fully modernized. Additional VF/BGC, ablation and historical checkpoint forward passes were not all rerun through the new Full-only wrapper. CPU-vs-GPU numerical parity was not established.

The local research review is same-family/provisional. Deterministic hash, syntax, metric and inference-replay checks are reported separately from scientific interpretation.

## Observed inference environment

The successful new wrapper replay used Windows, CUDA and the installed PyTorch build `2.12.0.dev20260408+cu128`. Other observed packages were NumPy 2.2.6, pandas 2.3.3 and scikit-learn 1.7.2. This is a record of the available replay environment, not a fabricated original training lockfile. The development PyTorch build may require a different installation source from a stable release.

`requirements.txt` lists broad minimum dependencies, not a validated universal version lock. Training, extraction and plotting scripts have additional dependencies visible in their imports, including language-model libraries and plotting tools.

## Original records versus release metadata

Original measurements and receipts were not rewritten. Newly created files are the release README/documentation, source/model indexes, archives, checksum lists, restore utility, prediction wrapper and release checks. The public model index and asset checksum list describe only the two final checkpoints and their published downloads. Broader experimental checkpoint archives remain unpublished. Hashes computed now attest to these current bytes, not to an absent historical attestation. A saved result file or a successful syntax check is not represented as proof of a fresh full training run.

Use the repository-root `restore_archive.py` for Windows long-path support.
The source ZIP's `release_entrypoints/` directory retains the initial packaging
snapshot; the repository-root utilities are the maintained release entrypoints.
