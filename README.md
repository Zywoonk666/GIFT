# GIFT

This is the codebase for **GIFT**, which is built and improved upon the GIMLET framework for molecule learning. GIFT is a unified graph-text model tailored and evaluated focusing deeply on **Few-Shot Learning** for downstream tasks.

## Installation

The fixed environment targets Linux x86_64, Python 3.7, PyTorch 1.9, and
CUDA 11.1. Create it with:

### Environment
```bash
conda env create -f environment.yml
conda activate env_gift
```

### Pre-download Hugging Face Assets

`run_experiments.sh` enables Hugging Face offline mode. On a clean machine,
download the required model and tokenizer once while network access is
available:

```bash
conda activate env_gift

python - <<'PY'
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

snapshot_download(
    repo_id="haitengzhao/gimlet",
    allow_patterns=["config.json", "generation_config.json", "pytorch_model.bin"],
)
AutoTokenizer.from_pretrained("t5-small")
print("Hugging Face assets are ready for offline use.")
PY
```

By default, these files are stored in `~/.cache/huggingface/`. The download
must be performed for the same user that will run the experiment. After the
cache is populated, no network connection is required by the experiment
script. If the target machine has no network access, copy the populated
Hugging Face cache from another machine before running.

## Dataset Download

The MoleculeNet datasets (including pcba, bace, hiv, muv, tox21, toxcast, bbbp, esol, lipo, and freesolv) can be downloaded manually or automatically upon the first run. 
To download manually:
```bash
wget http://snap.stanford.edu/gnn-pretrain/data/chem_dataset.zip
unzip chem_dataset.zip
mv dataset property_data
```

## Run Few-Shot Learning Experiments

We provide an integrated script to run the few-shot learning pipeline, which evaluates the model natively on downstream tasks.

To execute the reproduction of the few-shot learning downstream task, run:
```bash
bash run_experiments.sh
```

The script automates few-shot testing (e.g. on `bace`) leveraging the GIFT custom architecture. All evaluation results will be cached as CSV files in the `./cache/` directory.
