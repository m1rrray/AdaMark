# AdaMark

**AdaMark** is a framework for training and evaluating an adaptive digital image watermarking model.

The model's key feature is its **adaptivity**. A single `rf` parameter lets you smoothly tune the trade-off between watermark **imperceptibility** and **robustness**. The same trained model can therefore serve very different needs — from maximally hidden embedding to highly robust marks that survive strong attacks and compression.

## Features
* Train your own watermark embedding and extraction model.
* Evaluate robustness against a suite of simulated attacks.
* Configure everything through simple YAML config files.

## Project structure
```
src/adamark/
├── models/        # HidingNet, RevealingNet, rf-conditioned FiLM modulation
├── attacks/       # tampering and distortion attacks (train + eval)
├── data/          # datasets (FFHQ, CASIA1/2, Columbia) and transforms
├── losses/        # loss functions and watermark budget
├── training/      # distributed (DDP) training loop and checkpointing
├── evaluation/    # evaluation, attack suite, reporting (Excel + plots)
└── utils/         # seeding, DDP averaging, payload generation
configs/           # train_config.yaml, eval_config.yaml
tools/             # developer utilities (refactor equivalence checking)
train.py           # CLI entry point for training
evaluate.py        # CLI entry point for evaluation
```

## Installation
```bash
pip install -e .
```

## Training
```bash
torchrun --nproc_per_node=<num_gpus> train.py --config configs/train_config.yaml
```

## Evaluation
```bash
python evaluate.py --config configs/eval_config.yaml
```

---

This framework was developed as part of a graduation thesis:

🔗 **[Thesis (in Russian)](https://drive.google.com/file/d/1aHmX9R1BCK5CDxjCFg8FrHRdqmS2_XBC/view?usp=sharing)**

🔗 **[Download pretrained weights](#pretrained-weights-link-coming-soon)** (placeholder — coming soon)
