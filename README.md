# FedULIP 

This directory contains the FedULIP code and datasets.
Before using it, please clone the ULIP project to the FedULIP directory. Follow the instructions provided by the ULIP official to configure the environment, and finally place fedulip.py in the ULIP/models directory.

## Project Structure

- `base2new/`: Base-to-New scripts and configs.
- `cross_dataset/`: Cross-Dataset scripts and configs.
- `domain_abcd/`: Domain A/B/C/D scripts and visualization tools.
- `single_dataset/`: Single-Dataset scripts and configs.
- `ULIP/`: Backbone, training, and data-loading code (includes
  `pretrained_models/` and `data/` configs).
- `datasets/`: Datasets and processed data used in this project.

## Environment

Install dependencies:

```bash
cd ULIP
pip install -r requirements.txt
```

## Data

- Dataset paths are defined in `ULIP/data/*.yaml` and `ULIP/data/dataset_catalog.json`.
- Some datasets live under `ULIP/data/` (e.g., ModelNet40), others under
  `datasets/` (e.g., 3Dimage, ShapeNetCore).
- If you move datasets, update the corresponding `DATA_PATH` fields.

## Running Experiments

### Base-to-New

```bash
python base2new/fedulip_base2new.py \
  --pretrained pretrained_models/checkpoint_pointbert.pt \
  --output-dir outputs/base2new
```

### Cross-Dataset

```bash
python cross_dataset/fedulip_cross_dataset.py \
  --pretrained pretrained_models/checkpoint_pointbert.pt \
  --output-dir outputs/cross_dataset
```

### Domain A/B/C/D

```bash
python domain_abcd/fedulip_domain_abcd.py \
  --pretrained pretrained_models/checkpoint_pointbert.pt \
  --output-dir outputs/domain_abcd
```

### Single-Dataset

```bash
python single_dataset/fedulip_single_dataset.py \
  --pretrained pretrained_models/checkpoint_pointbert.pt \
  --output-dir outputs/single_dataset
```

## Common Arguments

Defaults match the paper settings; adjust as needed:

- `--num-clients`: number of clients
- `--rounds`: communication rounds
- `--local-epochs`: local training epochs
- `--batch-size` / `--eval-batch-size`: batch sizes
- `--supcon-weight`: SupCon weight
- `--devices`: GPU ids (e.g., `0,1`)

For a quick Domain A/B/C/D visualization, run `domain_abcd/preview_domain_abcd.py`.
