# ARGScope model construction

This page describes how ARGScope models are built and trained. To annotate sequences using the released models, follow the **[usage manual](../README.md)**.

ARGScope learns from paired protein and nucleotide representations, constructs a graph of ARG subtype prototypes, and transfers the learned information to separate protein and nucleotide students. At inference, each student requires only its own modality and predicts an ARG subtype alongside an ARG/non-ARG decision. A predefined subtype-to-type mapping provides the resistance type.

![ARGScope learning framework and inference](../assets/Figure1.png)

## Contents

- [Source code](#source-code)
- [Prepare training data](#prepare-training-data)
- [Training stages](#training-stages)
- [Run training](#run-training)
- [Ablation experiments](#ablation-experiments)
- [Training output](#training-output)
- [Other annotation tasks](#other-annotation-tasks)
- [Reproducibility](#reproducibility)

## Source code

The canonical training workflow is in [`training/arg_fixed80/`](../training/arg_fixed80/). The original source bundle, including split metadata and additional experimental workflows, is available as [`archive/ARGScope-source.zip`](../archive/ARGScope-source.zip).

Use the Python environment described in the [installation instructions](../README.md#installation). The canonical training workflow requires CUDA. Embedding extraction uses additional language-model dependencies and pretrained weights, which must be obtained separately.

## Prepare training data

Extract the source bundle into a separate directory:

```bash
python -m zipfile -e archive/ARGScope-source.zip source_bundle
```

The bundle supplies the structure file and fixed-split metadata under `source_bundle/data/`. Add the original paired embedding files to this data directory:

```text
source_bundle/data/
    all_ARG_nonARG_1to1_structure.txt
    split_80_20_grouped/
        split_contract.json
        split_manifest.csv
        population_ids.txt
        train_ids_subtype2.txt
        model_train_ids.txt
        validation_ids.txt
        test_ids_subtype2.txt
    ESM2_EMBEDDING/percent_100/
        full_1to1_esm2_mean.pt
    DNABERT2_embedding/percent_100/
        full_1to1_dnabert2_mean.pt
```

Embedding files contain `ids` and `embeddings`, with 1,280 features for protein and 768 for nucleotide inputs. Both files must cover the fixed sequence population. Full sequence FASTA files and embedding tensors are not distributed in the source bundle. See [DATA.md](DATA.md) for the data requirements.

The canonical population contains 22,870 sequences, partitioned as follows:

| Partition | Sequences |
|---|---:|
| Model training | 14,636 |
| Validation | 3,660 |
| Test | 4,574 |

All stages read the same saved split. Training and test partitions each contain 820 ARG subtypes spanning 23 resistance types, together with non-ARG sequences. The split fingerprint is `6d2d3cf7ad252f255dedf4e4bbeae693f5ea0c5112f39daec331aaa975d93991`.

The scripts verify the supplied split, sequence population and structure file. They do not create a replacement split for a different dataset. Normalization statistics and raw subtype prototypes are computed from model-training sequences. Validation data are used by the existing selection procedures.

## Training stages

| Stage | Source file | Purpose |
|---|---|---|
| 1. Multimodal fusion | [`01_gated_fusion.py`](../training/arg_fixed80/core/01_gated_fusion.py) | Project protein and nucleotide embeddings and learn adaptive feature-wise fusion |
| 2. Raw prototypes | [`02_raw_prototypes.py`](../training/arg_fixed80/core/02_raw_prototypes.py) | Construct subtype centroids from fused model-training ARG representations |
| 3. Graph refinement | [`03_refine_gat.py`](../training/arg_fixed80/core/03_refine_gat.py) | Refine subtype prototypes using a mutual nearest-neighbor graph and residual graph attention |
| 4. Protein teacher and student | [`04_protein_student.py`](../training/arg_fixed80/core/04_protein_student.py) | Train the protein-branch teacher and transfer supervision to ARGScope-AA |
| 5. Nucleotide teacher and student | [`05_dna_student.py`](../training/arg_fixed80/core/05_dna_student.py) | Train the nucleotide-branch teacher and transfer supervision to ARGScope-NT |

The graph-refined prototypes guide subtype-head initialization and anchoring. Student training combines classification supervision with prediction distillation, feature alignment, prototype alignment and head anchoring. The graph and the other input modality are not needed when using a trained student for prediction.

## Run training

Run the commands from the repository root. First check the data, fixed split and CUDA environment without training a model:

```bash
python training/arg_fixed80/run_pipeline.py --data-root source_bundle/data --output-root results/training --variant full --preflight-only
```

After `PREFLIGHT PASS`, run the Full workflow:

```bash
python training/arg_fixed80/run_pipeline.py --data-root source_bundle/data --output-root results/training --variant full
```

This executes fusion, raw prototype construction, graph refinement, and both teacher-student workflows in order. To train only one student branch, append `--modality protein` or `--modality dna`. The required upstream stages are still included.

### Training options

| Option | Description |
|---|---|
| `--data-root` | Directory containing the structure, split and embedding files |
| `--output-root` | Separate directory for this run's outputs and logs |
| `--variant` | `full`, `wo_gate`, `wo_graph`, `wo_transfer` or `all` |
| `--modality` | `both` by default, or `protein` or `dna` |
| `--preflight-only` | Check inputs and environment, then exit |
| `--smoke-test` | One-epoch integration check in a separate run; not a scientific result |

Completed stages are reused only when their saved completion records and hashes agree. An incomplete or unrecognized stage directory causes the workflow to stop. Preserve that directory under another name before restarting the stage. The workflow does not resume from an interrupted epoch.

## Ablation experiments

The same suite provides the following variants:

| Variant | Change from Full |
|---|---|
| `wo_gate` | Replace the learned gate with equal protein/nucleotide weights while retaining the projections and fusion normalization |
| `wo_graph` | Use raw subtype prototypes instead of graph-refined prototypes |
| `wo_transfer` | Train students with classification supervision, with teacher optimization and transfer losses disabled |

Run Full and all three ablations sequentially:

```bash
python training/arg_fixed80/run_pipeline.py --data-root source_bundle/data --output-root results/training --variant all
```

After all eight student models, covering four variants and two modalities, have finished, summarize their predictions:

```bash
python training/arg_fixed80/summarize_results.py --data-root source_bundle/data --output-root results/training
```

The summary checks that all models use the same 4,574 test IDs. It reports binary ARG-positive metrics and type/subtype macro, weighted and micro metrics. Non-ARG sequences remain in the type and subtype evaluations.

## Training output

```text
results/training/
    full/
        fusion/
        raw/
        graph/
        protein/
        dna/
    wo_gate/       # when requested
    wo_graph/      # when requested
    wo_transfer/   # when requested
    logs/
    metrics_*/     # after summarizing all variants
```

The student directories contain checkpoints, `test_sequence_predictions.csv`, split manifests and training records. Stage completion records track the code and input hashes used by each run.

The public [`inference/predict.py`](../inference/predict.py) wrapper accepts only the two byte-verified released Full checkpoints. A newly trained checkpoint has a different hash and cannot be substituted into that wrapper unchanged. Use the saved-model application/evaluation workflows in `source_bundle/evaluation/arg_saved_models/` as the starting point for a corresponding inference implementation.

## Other annotation tasks

Original task-specific sources are available for [virulence factor annotation](../training/vf_original/), [BGC-associated sequence annotation](../training/bgc_original/) and the [standardized BGC workflow](../training/bgc_standardized/). These tasks require their own data preparation and retraining. The public model download contains the ARGScope-AA and ARGScope-NT checkpoints only.

## Reproducibility

The original released checkpoint bytes were preserved. Each was replayed on its original 4,574 test embeddings, with binary, subtype and type predictions matching the saved records. This check did not retrain the models. The source archive contains multiple experimental versions, so use the canonical fixed80 workflow and its matching metadata together.

- [Replay verification](../metadata/INFERENCE_REPLAY.json)
- [Model index](../metadata/MODEL_INDEX.csv)
- [Source index](../metadata/SOURCE_INDEX.csv)
- [Environment and release checks](REPRODUCIBILITY.md)
- [Data requirements](DATA.md)
- [Licensing status](LICENSE_STATUS.md)

Questions and bug reports can be submitted through [GitHub issues](https://github.com/JiePan-code/ARGScope/issues) or sent to [pan.jie@ntu.edu.sg](mailto:pan.jie@ntu.edu.sg).
