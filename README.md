# AdaMark

**AdaMark** is a framework for training and evaluating an adaptive semi-fragile image watermarking model.

The model's key feature is its **adaptivity**. A single `rf` parameter (the robustness factor) lets you smoothly tune the trade-off between watermark **imperceptibility** and **robustness** at inference time, without retraining. The same trained model can therefore serve very different needs — from maximally hidden embedding to highly robust marks that survive strong attacks and compression.

This is the reference implementation for the paper *"AdaMark: Adaptive semi-fragile watermarking with controlled robustness"* (submitted to Neurocomputing).

## Features
* Train your own watermark embedding and extraction model.
* Evaluate robustness against a suite of simulated attacks.
* Analyse how watermark energy is distributed across frequency bands.
* Reproduce the ablation study through config switches.
* Configure everything through simple YAML config files.

## Project structure
```
src/adamark/
├── models/        # HidingNet, RevealingNet, rf-conditioned FiLM modulation
├── attacks/       # tampering and distortion attacks, for training and evaluation
├── data/          # datasets: FFHQ, CASIA1/2, Columbia, and transforms
├── losses/        # loss functions and watermark budget
├── training/      # distributed DDP training loop and checkpointing
├── evaluation/    # evaluation, attack suite, spectral analysis, reporting
└── utils/         # seeding, DDP averaging, payload generation
configs/            # train_config.yaml, eval_config.yaml
tools/              # developer utilities, e.g. numerical equivalence checking
train.py            # CLI entry point for training
evaluate.py         # CLI entry point for evaluation
analyze_spectrum.py # CLI entry point for the frequency analysis
```

## Installation
```bash
pip install -e .
```

## Training
```bash
torchrun --nproc_per_node=<num_gpus> train.py --config configs/train_config.yaml
```

The published model was trained for 150 epochs with batch size 64, an initial
learning rate of 3.5e-4 with cosine annealing, on two NVIDIA A100 40GB GPUs.
The composite objective uses `lambda_h = 1` (fixed inside `HidingLoss`),
`revealing_loss_weight = 3` and `lambda_fft = 3`.

## Evaluation
```bash
python evaluate.py --config configs/eval_config.yaml
```

Point `test_datasets` in `configs/eval_config.yaml` at your local copies of CASIA1,
CASIA2 and Columbia first — it ships with the entries commented out. Results are
written to `results/` as an `.xlsx` table plus per-attack plots.

To evaluate a single manipulation type instead of both, set `tamper_filter` to
`splicing` or `copymove`.

## Frequency analysis
```bash
python analyze_spectrum.py --config configs/eval_config.yaml
```

Reports the share of watermark energy per frequency band for each `rf` value. The
radial coordinate matches the one used by `FFTLoss`, so the numbers are directly
comparable to the spectral penalty applied during training. Band edges default to
six equal-width annuli and can be overridden:

```bash
python analyze_spectrum.py --config configs/eval_config.yaml --bands 0 0.15 0.3 0.45 0.6 0.8 inf
```

## Ablation study

The three components of the adaptive mechanism are switched independently in
`configs/train_config.yaml`. All `True` reproduces the full model:

| Flag | `False` disables |
|---|---|
| `use_film` | rf conditioning; FiLM layers become identity but keep their parameters, so checkpoints stay loadable |
| `use_fft_loss` | the spectral penalty; the FFT term drops out of the objective |
| `use_budget` | RMS budgeting; the raw `tanh` watermark is added to the cover without centering or rescaling |

Flip a flag, point `save_dir` at a fresh directory, and retrain. When evaluating an
ablated checkpoint, set the matching `use_film` / `use_budget` in
`configs/eval_config.yaml` — otherwise the embedding path will not match the one
the weights were trained for.

## Comparison with other methods

The paper compares AdaMark against TruFor and MVSS-Net. Their code and weights are
not vendored here; reproduce the comparison by running the official
implementations and applying the same protocol:

* use the authors' released pretrained weights, without fine-tuning;
* note the resolution difference — both reference methods operate at 512x512, while
  AdaMark operates at 256x256;
* apply distortions with the same parameters as `build_eval_attacks`
  (`src/adamark/evaluation/attack_suite.py`);
* score with pixel-level ROC AUC over the same images and ground-truth masks.

---

🔗 **[Download pretrained weights](#pretrained-weights-link-coming-soon)** (placeholder — coming soon)

Background reading: the earlier graduation thesis this framework grew out of is
available [in Russian](https://drive.google.com/file/d/1aHmX9R1BCK5CDxjCFg8FrHRdqmS2_XBC/view?usp=sharing).
