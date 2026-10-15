# Data and inputs

This code/model snapshot does not redistribute the complete reference sequence databases or the ESM-2/DNABERT-2 backbone weights. Obtain those from their original providers under their respective terms. No new biological sequences or experimental phenotypes were generated for the release.

## Canonical fixed80 training layout

The original suite expects a data directory containing:

```text
all_ARG_nonARG_1to1_structure.txt
split_80_20_grouped/
    split_contract.json
    split_manifest.csv
    model_train_ids.txt
    validation_ids.txt
    test_ids_subtype2.txt
ESM2_EMBEDDING/percent_100/full_1to1_esm2_mean.pt
DNABERT2_embedding/percent_100/full_1to1_dnabert2_mean.pt
```

Existing label/split metadata are included in the source bundle's `data/` directory. Embedding tensors and sequence FASTA are not included. A training directory copied from another experiment is not an equivalent replacement.

The fixed population contains 22,870 sequences: 14,636 model-training, 3,660 validation and 4,574 test records. The authoritative contract and original source-file hashes are included. The checkpoint classifier has 820 ARG subtypes, with non-ARG handled by its binary head.

## Prediction input

`inference/predict.py` expects a trusted `.pt` dictionary:

```python
{"ids": ["sequence_1", "sequence_2"], "embeddings": tensor_of_shape_N_by_D}
```

D is 1,280 for the protein branch or 768 for the nucleotide branch. Preserve original extraction definitions. Do not pass fused representations, standardized embeddings, or embeddings from a different pooling strategy as if they were equivalent.

## Additional experiments

VF data derive from VFDB and BGC-associated data from MIBiG, with non-target candidates from the corresponding project preparation. Source annotations are reference labels, not experimental confirmation of each predicted sequence function. BGC annotation is sequence-level assignment, not complete-cluster boundary detection. Original and standardized BGC releases have separate label handling and model schemas.

No completeness claim is made for redistribution of every intermediate input or external database. `metadata/MODEL_INDEX.csv` and `metadata/SOURCE_INDEX.csv` record the actual supplied artifacts.
