# ARGScope

ARGScope annotates antibiotic resistance genes (ARGs) from protein or nucleotide representations. It provides two independently usable models, **ARGScope-AA** for protein inputs and **ARGScope-NT** for nucleotide inputs. Each model predicts ARG/non-ARG status and an ARG subtype. Resistance types are obtained from the predicted subtypes using the saved label mapping.

This page is the **usage manual**. For the learning framework, training data layout and training commands, see **[ARGScope model construction](docs/DEVELOPMENT.md)**.

## Contents

- [Installation](#installation)
- [Download pretrained models](#download-pretrained-models)
- [Prepare input](#prepare-input)
- [Example](#example)
- [Output](#output)
- [Model construction](#model-construction)
- [Notes](#notes)
- [Contact](#contact)

## Installation

Clone this repository:

```bash
git clone https://github.com/JiePan-code/ARGScope.git
cd ARGScope
```

We suggest creating a separate Conda environment:

```bash
conda create -n argscope python=3.11 -y
conda activate argscope
python -m pip install -r requirements.txt
```

For GPU prediction, install a CUDA-enabled PyTorch build appropriate for your system. The dependencies in `requirements.txt` are minimum requirements. The environment used for the checkpoint replay is recorded in [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

Check the available prediction options:

```bash
python inference/predict.py --help
```

Run all commands below from the repository root.

## Download pretrained models

Download **[ARGScope-pretrained-full.zip](https://github.com/JiePan-code/ARGScope/releases/download/v1.0.0/ARGScope-pretrained-full.zip)** from the [v1.0.0 release](https://github.com/JiePan-code/ARGScope/releases/tag/v1.0.0), save it in the repository root, and extract it:

```bash
python -m zipfile -e ARGScope-pretrained-full.zip .
```

The two checkpoints will be located at:

```text
models/arg_fixed80/full/
    ARGScope-AA.pt
    ARGScope-NT.pt
```

These are the final Full models used for the manuscript's main full-length evaluation. They contain the student weights, training normalization statistics, subtype labels and subtype-to-type mapping. Model file checksums are listed in [MODEL_INDEX.csv](metadata/MODEL_INDEX.csv).

## Prepare input

**The current prediction script accepts precomputed embeddings, not raw FASTA or FASTQ files.** Prepare embeddings with the corresponding pretrained language model:

| Model | Input representation | Dimensions |
|---|---|---:|
| ARGScope-AA | ESM-2 `esm2_t33_650M_UR50D` protein embeddings | 1,280 |
| ARGScope-NT | DNABERT-2 nucleotide embeddings | 768 |

Save each input as a PyTorch `.pt` dictionary with these two entries:

| Entry | Required content |
|---|---|
| `ids` | A list of unique, nonempty sequence ID strings |
| `embeddings` | A finite tensor of shape `N x 1280` for AA or `N x 768` for NT, in the same order as `ids` |

If your extraction workflow already provides `sequence_ids` and `embeddings`, package them as follows:

```python
import torch

torch.save(
    {"ids": sequence_ids, "embeddings": embeddings.cpu().float()},
    "protein_embeddings.pt",
)
```

Use `nucleotide_embeddings.pt` as the output name for the NT input. The example above packages existing embeddings; it does not extract them from sequences. Preserve the original extraction, pooling and sequence-handling settings. For AA, use mean layer-33 representations of real residues, excluding BOS/EOS and retaining at most the first 1,022 residues. Do not standardize the embeddings before prediction, because the saved model applies its training normalization internally. See [DATA.md](docs/DATA.md) for input details.

## Example

### Protein annotation

With the AA checkpoint and `protein_embeddings.pt` prepared:

```bash
python inference/predict.py --modality protein --checkpoint models/arg_fixed80/full/ARGScope-AA.pt --embeddings protein_embeddings.pt --output results/predictions_AA.csv
```

### Nucleotide annotation

With the NT checkpoint and `nucleotide_embeddings.pt` prepared:

```bash
python inference/predict.py --modality dna --checkpoint models/arg_fixed80/full/ARGScope-NT.pt --embeddings nucleotide_embeddings.pt --output results/predictions_NT.csv
```

Only the selected modality is required for prediction. Paired protein and nucleotide inputs are used during model training.

### Options

| Option | Description |
|---|---|
| `--modality` | `protein` or `dna` |
| `--checkpoint` | Path to the corresponding released checkpoint |
| `--embeddings` | Path to the input `.pt` dictionary |
| `--output` | Path to a new output CSV file |
| `--device` | `cuda` by default, or `cpu` |
| `--batch-size` | Optional positive integer overriding the saved evaluation batch size |

For CPU prediction, append `--device cpu` to the command. CPU execution is implemented, but agreement with the original GPU results has not been verified. Existing output files are not overwritten.

## Output

The output CSV contains one row per input sequence:

| Column | Description |
|---|---|
| `sequence_id` | Input sequence ID |
| `predicted_ARG` | Binary decision, with `1` for ARG and `0` for non-ARG |
| `predicted_subtype` | Predicted ARG subtype, or `nonARG` |
| `predicted_type` | Resistance type mapped from the predicted subtype, or `nonARG` |
| `ARG_probability` | Probability from the binary ARG head |
| `ARG_subtype_max_probability` | Maximum probability from the ARG subtype head |
| `ARG_subtype_before_binary_gate` | Highest-scoring ARG subtype before applying the binary decision |
| `binary_threshold` | Saved ARG probability threshold, `0.80` |

The binary and subtype heads run in parallel. Predictions with an ARG probability below `0.80` are reported as `nonARG`. For ARG predictions, the subtype is retained and mapped to its resistance type.

## Model construction

The separate **[model construction guide](docs/DEVELOPMENT.md)** describes multimodal fusion, subtype prototype construction, graph refinement, teacher-student learning and the fixed-split training commands. The [training source](training/arg_fixed80/) can be browsed directly on GitHub.

## Notes

- Prediction does not require retraining the model or providing the other modality.
- The released wrapper checks the checkpoint hashes and supports the two published Full checkpoints. Newly trained or alternative-task checkpoints need their corresponding inference workflow.
- Load only trusted PyTorch model and embedding files. Full sequence databases, language-model weights and bulk embeddings are not included.
- The two released checkpoints were replayed on their original 4,574 test embeddings, matching the saved predictions. Details are in [INFERENCE_REPLAY.json](metadata/INFERENCE_REPLAY.json).
- Data requirements, release checks and reuse terms are described in [DATA.md](docs/DATA.md), [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) and [LICENSE_STATUS.md](docs/LICENSE_STATUS.md).

## Contact

For questions or bug reports, please open a [GitHub issue](https://github.com/JiePan-code/ARGScope/issues) or contact [pan.jie@ntu.edu.sg](mailto:pan.jie@ntu.edu.sg). Please include the command, error message and software versions when reporting a problem.
