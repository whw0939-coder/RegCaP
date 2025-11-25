# RegCaP: Region-Centric, Causal-Inspired Vulnerability Detection with Distributional Priors

# Reproduction Guide

RegCaP detects software vulnerabilities via **region-level causal modeling** that integrates structural partitioning, regional distributional priors, and intervention-based robustness tests.
This repository provides code and instructions to reproduce all experiments reported in the paper.

---

## 1 Project structure

```
RegCaP/
├── joern/                         # Joern CLI
├── dataset/                       
│   ├── FFMPeg+Qemu.pkl
│   ├── BigVul.pkl
│   ├── DiverseVul.pkl
│   └── Reveal.pkl
├── CWE/                           
│   ├── BigVul/
│   │   ├── CWE-119.pkl
│   │   ├── CWE-… .pkl
│   │   └── …
│   └── DiverseVul/
│       ├── CWE-… .pkl
│       └── …
├── model/                         # trained checkpoints
│   ├── FFMPeg+Qemu/
│   ├── BigVul/
│   ├── DiverseVul/
│   ├── BigVul-CWE/
│   └── DiverseVul-CWE/
├── split_lists/                   # train/val/test splits (FFMPeg+Qemu)
├── train_stats/                   # precomputed token/API stats for training set
├── delta_is/                      # ΔF1 & IS results/figures (token OR API)
├── delta_is_both/                 # ΔF1 & IS results/figures (token AND API)
├── main.py                        # train / evaluate
├── model.py                       # network definition
├── training.py                    # training loop
├── W2V-training.py                # training W2V 
├── data_preprocessing.py          # dataset-level preprocessing
├── data_preprocessing-CWE.py      # CWE-level preprocessing
├── data_loading.py
├── stats_train_tokens_apis.py     # token/API distribution on training set
├── perturb_testset.py             # perturb tokens or APIs separately
├── perturb_both.py                # jointly perturb tokens & APIs
├── calc_delta_is.py               # compute ΔF1 & IS under perturbations
├── inspect_test_details.py        # detailed causal-region inspection
└── export_ast_ccd_from_dot.sc     # joern extraction scripts
```

---

## 2 Environment and Dependencies

| Item           | Tested version         |
| -------------- | ---------------------- |
| Python         | 3.8.2                  |
| PyTorch + CUDA | 2.4.1 / 12.1           |
| Joern CLI      | 2.0.86                 |
| Others         | see `requirements.txt` |

```bash
cd RegCaP
pip install -r requirements.txt
```

---

## 3 Dataset Preparation

### Core datasets & support files

* Datasets: **FFMPeg+Qemu**, **BigVul**, **DiverseVul**, **Reveal**
* Word2Vec models: `dataset/*/W2V/`
* Joern CLI: `joern/`

> **Data & tools (Zenodo Sandbox)**
>
> Due to repository size limits, the raw corpora and Joern binaries are hosted on Zenodo Sandbox:
> - `FFMPeg+Qemu.zip`
> - `BigVul.zip`
> - `DiverseVul.zip`
> - `Reveal.zip`
> - `joern.zip`
>
> All files are available under the anonymous record:
> https://sandbox.zenodo.org/records/401848
>
> After download & extraction (from the project root):
>
> 1. Extract `joern.zip` so that the Joern CLI folder appears as:
>    `./joern/`
> 2. Extract each dataset archive (`FFMPeg+Qemu.zip`, `BigVul.zip`, `DiverseVul.zip`, `Reveal.zip`).  
>    This will populate:
>    - `./dataset/FFMPeg+Qemu.pkl`
>    - `./dataset/BigVul.pkl`
>    - `./dataset/DiverseVul.pkl`
>    - `./dataset/Reveal.pkl`
> 3. If you want to fully regenerate the PKLs from raw `.c` files, follow the
>    **AST generation with Joern** and **data_preprocessing** steps described
>    in §3; otherwise you can directly use the provided `.pkl` files under `dataset/`.


### AST/CFG/pdg generation with Joern

```bash
# Add Joern CLI to PATH
export PATH="$PATH:/your-path/RegCaP/joern/joern-cli"

# Run extraction for all datasets
joern --script export_ast_ccd_from_dot.sc --param srcBaseArg="/path/dataset/*/c/"   --param outBaseArg="/path/dataset/*/js/"
```

This scans every `.c` file under `dataset/*/c/` and writes AST JSONs to `dataset/*/js/`.

---

## 4 Experiment Reproduction by Research Question (RQ1–RQ5)

| RQ  | Goal                                             | Main scripts                                                                                         |
| --- | ------------------------------------------------ | ---------------------------------------------------------------------------------------------------- |
| RQ1 | Function-level training & baseline comparison    | `data_preprocessing.py`, `main.py`                                                                   |
| RQ2 | Cross-dataset & CWE-level generalization         | `data_preprocessing.py`, `data_preprocessing-CWE.py`, `main.py`                                      |
| RQ3 | Robustness to semantics-preserving perturbations | `stats_train_tokens_apis.py`, `perturb_testset.py`, `perturb_both.py`, `calc_delta_is.py`, `main.py` |
| RQ4 | Ablation studies                                 | edit `model.py`, then `main.py`                                                                      |
| RQ5 | Region-level causal inspection                   | `data_preprocessing.py`, `inspect_test_details.py`, `main.py`                                        |

---

### RQ1 — Function-level training & evaluation

```bash
# Example: FFMPeg+Qemu
python data_preprocessing.py --dataset FFMPeg+Qemu
# → generates FFMPeg+Qemu-12-Score.pkl

python main.py \
  --gpus 0 \
  --data_path "FFMPeg+Qemu-12-Score.pkl"
```

Repeat with `BigVul` and `DiverseVul` to reproduce results on the other datasets.

---

### RQ2 — Cross-dataset and CWE-level generalization

#### (1) Cross-dataset transfer (train on one dataset, test on another)

```bash
# Prepare Reveal
python data_preprocessing.py --dataset Reveal
# → generates Reveal-12-Score.pkl

# Example: use a checkpoint trained on FFMPeg+Qemu, test on Reveal
python main.py \
  --gpus 0 \
  --cls_model_path ./model/FFMPeg+Qemu/checkpoints/ \
  --data_path ./Reveal-12-Score.pkl \
  --cross_data_path ./Reveal-12-Score.pkl
```

> We also provide checkpoints trained on `BigVul/` and `DiverseVul/` under `model/`.

#### (2) CWE-level training & evaluation

```bash
# Generate per-CWE PKLs
python data_preprocessing-CWE.py --dataset BigVul

# Train in CWE mode (produces overall results)
python main.py \
  --gpus 0 \
  --cwe_mode \
  --data_path "CWE/BigVul/"
```

Evaluate a single CWE:

```bash
python main.py \
  --gpus 0 \
  --data_path "CWE/BigVul/CWE-119.pkl" \
  --cls_model_path checkpoints/
```

---

### RQ3 — Robustness under token/API perturbations

1. Analyze training-set distributions (uses split_lists/ automatically):

```bash
python stats_train_tokens_apis.py
```
Outputs distribution files under train_stats/.

2. Create perturbed test sets:

```bash
python perturb_testset.py   # perturb tokens OR APIs
python perturb_both.py      # perturb tokens AND APIs
```

3. Re-run Joern on perturbed code and preprocess to PKL, then evaluate:

```bash
python main.py \
  --gpus 0 \
  --cls_model_path ./model/FFMPeg+Qemu/checkpoints/ \
  --data_path <perturbed.pkl> \
  --cross_data_path <perturbed.pkl>
```

4. Compute ΔF1 & IS and aggregate plots:

```bash
python calc_delta_is.py
# → outputs ./delta_is/delta_is_summary.json and figures
#   (For joint perturbations, corresponding outputs go to ./delta_is_both/)
```

---

### RQ4 — Ablation studies

Comment out the target modules in `model.py` (e.g., specific encoders, priors, or gating blocks), then retrain/evaluate:

```bash
python main.py \
  --gpus 0 \
  --data_path "FFMPeg+Qemu-12-Score.pkl"
```

The relative drops (F1 Score) quantify each component’s contribution.

---

### RQ5 — Region-level causal inspection

Prepare a folder with code for inspection:

```
Location/
└── c/
    ├── 0_1.c
    ├── 1_0.c
    └── ...
```

Each file is named `base_label.c` (label `0/1` can be dummy if unknown).

Run Joern on `Location/c`, preprocess, then evaluate with a trained model:

```bash
joern --script export_ast_ccd_from_dot.sc --param srcBaseArg="Location/c/"   --param outBaseArg="Location/js/"

python W2V-training.py --json-dir "Location/js/" --out-dir "Location/W2V/" --model-name "Location-128-20.wordvectors"

python data_preprocessing.py --dataset Location
# → Location-12-Score.pkl

python main.py \
  --gpus 0 \
  --cls_model_path ./model/FFMPeg+Qemu/checkpoints/ \
  --data_path ./Location-12-Score.pkl \
  --cross_data_path ./Location-12-Score.pkl
```

Find per-region predictions under:

```
./runs_eval/vul_multi_obj_eval/version_xxx/test_dump/
```

Produce detailed inspection for a specific file:

```bash
python inspect_test_details.py \
  --pkl "<path-to-test_dump-file>" \
  --name "0_1"
```

---

### Custom dataset testing & causal analysis (general)

To test on other projects:

1. Ensure each source file is named `base_label.c` (label optional).
2. Run Joern extraction → `data_preprocessing.py` to build `<name>-12-Score.pkl`.
3. Apply a trained checkpoint with `main.py` (`--cls_model_path`).
   `vulnerable_lines.json` is **not** required for testing.
   For causal inspection, follow the RQ5 workflow (labels can be dummy).

---
