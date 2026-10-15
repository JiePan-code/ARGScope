# ARGScope

Graph-guided multimodal fusion and distillation for hierarchical antibiotic resistance gene annotation.

ARGScope combines paired protein and nucleotide representations during training, refines subtype prototypes using a mutual nearest-neighbor graph, and transfers supervision to independently deployable protein and nucleotide students. Binary and subtype heads run in parallel. The binary decision determines whether a subtype annotation is retained, and a predefined mapping supplies the resistance type.

![ARGScope framework](Figure1.png)

## Download

- **[Original code bundle](ARGScope-source.zip)** — structured training/evaluation workflows, fixed-split metadata, and a checksum-indexed archive of historical experimental and plotting sources.
- **[Current manuscript models](https://github.com/JiePan-code/ARGScope/releases/tag/v1.0.0)** — the final Full ARGScope-AA and ARGScope-NT checkpoints only.
- **[Model index](MODEL_INDEX.csv)** and **[release asset checksums](RELEASE_ASSETS.csv)** — original relative paths, packaged locations, sizes and SHA-256 identifiers for the published files.

The supplied Figure 1 is included without image modification. The model download contains only the two final ARGScope checkpoints used for the manuscript's main full-length evaluation. Historical, smoke-test, search-candidate, ablation and alternative-task checkpoints are not included in the public model release. Existing checkpoint bytes and reported measurements have not been replaced by newly trained models or reconstructed values.

## Protein-only or nucleotide-only prediction

The root `predict.py` entry point accepts **precomputed embeddings**, not raw FASTA. It is a release wrapper added on 2026-10-15. `student_models.py` is an exact copy of the existing saved-model inference definitions. It uses saved normalization statistics, subtype labels, classifier weights and the binary probability threshold of 0.80. Inputs must not be standardized a second time.

1. Download and extract `ARGScope-pretrained-full.zip` from the release.
2. Prepare a trusted PyTorch dictionary containing `ids` (unique strings) and `embeddings` (an N-by-D tensor).
3. Run one branch:

```bash
python predict.py --modality protein --checkpoint models/arg_fixed80/full/ARGScope-AA.pt --embeddings protein_embeddings.pt --output predictions_AA.csv

python predict.py --modality dna --checkpoint models/arg_fixed80/full/ARGScope-NT.pt --embeddings nucleotide_embeddings.pt --output predictions_NT.csv
```

Protein inputs are 1,280-dimensional ESM-2 `esm2_t33_650M_UR50D` representations. Nucleotide inputs are 768-dimensional DNABERT-2 representations. Extraction, pooling and sequence handling must match the original workflow. The checkpoint describes protein pooling as the mean of real-residue layer-33 representations, excluding BOS/EOS and retaining the first 1,022 residues. Original extraction and preprocessing scripts are retained in the source archive; older CLS-based and mean-pooled pipelines must not be interchanged.

CUDA is the replay-verified path. `--device cpu` is implemented but its numerical agreement with the original reported GPU results has not been validated. Loading uses `weights_only=True`; load only trusted model and embedding files.

## Training and reproduction

Extract `ARGScope-source.zip`. The canonical fixed80 workflow is in `training/arg_fixed80/`. Provide the original data layout described in [DATA.md](DATA.md), then run:

```bash
python training/arg_fixed80/run_pipeline.py --data-root /path/to/data --output-root /path/to/new_results --variant full
```

The original suite also exposes `wo_gate`, `wo_graph`, `wo_transfer`, and `all`. Its source files and original run receipts are retained. Training has **not** been repeated for this release. Historic Windows/HPC paths in original scripts need local configuration. The new prediction wrapper supports the two published Full checkpoints, not every checkpoint schema represented in the source archive.

The structured source bundle includes:

| Directory | Contents |
|---|---|
| `training/arg_fixed80/` | Frozen canonical fusion, prototype, graph, teacher/student and ablation scripts |
| `evaluation/arg_saved_models/` | Original saved-model and fragment application/evaluation scripts |
| `data/` | Existing fixed-split IDs, label metadata and split contracts; no sequence FASTA |
| `training/vf_original/` | Original VF workflow |
| `training/bgc_original/` | Original BGC workflow |
| `training/bgc_standardized/` | Separate standardized BGC workflow |
| `historical/novel_gene/` | Original historical experiment scripts |
| `embeddings/` | Identified embedding-extraction sources |
| `archive/` | Other original sources, deduplicated by bytes and mapped by `CODE_INDEX.csv` |

Use `restore_archive.py` to restore archived sources under their original relative names. Archived scripts are not represented as uniformly portable or independently replay-tested.

## Verified Full-model results

The saved full-length test population contains 4,574 records, including ARG and non-ARG sequences. Values below are rounded from the original saved predictions, not read back from figure labels.

| Model | Binary ACC | Subtype ACC | Subtype weighted F1 |
|---|---:|---:|---:|
| ARGScope-AA | 0.9976 | 0.9434 | 0.9313 |
| ARGScope-NT | 0.9746 | 0.9257 | 0.9092 |

For release verification, both original Full checkpoints were run on their original 4,574 test embeddings. All binary, type and subtype predictions matched the saved tables, with maximum absolute probability difference **0.0**. See [INFERENCE_REPLAY.json](INFERENCE_REPLAY.json). This is forward-inference replay, not an independent validation dataset or a fresh training run.

## Versioning and scope

The fixed80 split fingerprint is `6d2d3cf7ad252f255dedf4e4bbeae693f5ea0c5112f39daec331aaa975d93991`. The downloadable model version contains only the final ARGScope-AA and ARGScope-NT weights identified in `MODEL_INDEX.csv`. Other experiment sources are retained for traceability but do not imply that their model weights are distributed. No universal superiority, statistical significance or prospective deployment claim is added by this repository.

Original recorded results remain unchanged in the local research archive. The public `INFERENCE_REPLAY.json` records the two final checkpoint replay checks. The source archive contains multiple experiment versions, which should not be merged solely because filenames look similar.

## Dependencies, data and licensing

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for observed runtime and verification scope, [DATA.md](DATA.md) for data expectations, and [LICENSE_STATUS.md](LICENSE_STATUS.md) for licensing status. Upstream language-model weights, full sequence databases, bulk precomputed embeddings, temporary document-editing utilities and files with potential embedded credentials are not redistributed. Exclusions are indexed in `OMITTED_FILES.csv` and `OMITTED_MODEL_DATA.csv`.

Contact: pan.jie@ntu.edu.sg
