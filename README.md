```markdown
# VORQEN

> A small language model built from scratch by a middle schooler.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

## Introduction

**VORQEN** is a language model project built from scratch by a middle school student.

Without access to enterprise GPUs, this project aims to train small language models
using only **free computing resources** (Google Colab, Kaggle), and gradually
scale them up over time.

Building something like GPT requires hundreds of millions of dollars and tens of
thousands of GPUs. But **"starting small and growing steadily"** is something
anyone can do. This project documents that journey in code.

## Goals

- **Short-term**: Fully understand Transformers + implement a mini GPT
- **Mid-term**: Train a 10M–50M parameter model on Korean data
- **Long-term**: Fine-tune open-source small models into a practical chatbot

## Roadmap

- [x] **Phase 0**: GitHub repository setup
- [x] **Phase 1**: Mini GPT — Transformer implemented from scratch ✅
- [ ] **Phase 2**: Korean VORQEN v0.1 (10M) — trained on Korean data
- [ ] **Phase 3**: VORQEN Chatbot — LoRA fine-tuning + Hugging Face deployment
- [ ] **Phase 4**: VORQEN Pro — domain-specialized model

## Milestones

### 2026-09-21 — First successful training 🚀

- **Pipeline complete**: Tokenizer → Model → Training → Save → Generation
- **Model**: `VORQEN-tiny` (~**0.5M parameters**)
- **Environment**: Google Colab T4 GPU
- **Training**: 5,000 steps / 198 seconds
- **Result**: loss **3.0 → 0.003**

**First generated sentence:**
```

[Input: 안녕]
안녕하세요. 저는 VORQEN입니다. 인공지능 언어모델입니다.

```

> At this stage, the model was trained on a small repeated corpus,
> so it essentially "memorized" the data. Real Korean language
> training begins in the next phase.

## Architecture

**VORQEN** is a GPT-style **Decoder-only Transformer**.

```

Input text
↓
[CharTokenizer]  chars → ids
↓
[Embedding]      token embedding + positional embedding
↓
[Transformer × N]
├── Multi-Head Self-Attention (causal)
└── Feed-Forward Network (GELU)
↓
[LayerNorm]
↓
[LM Head]        next-token prediction
↓
Output text

```

## Project Structure

```

VORQEN/
├── vorqen/                    # Core package
│   ├── models/                # Model definitions
│   │   ├── layers.py          # Attention, FFN, Block
│   │   └── transformer.py     # VORQEN main model
│   ├── tokenizers/            # Tokenizers
│   │   └── char.py            # Character-level
│   ├── data/                  # Data loading
│   │   └── dataset.py         # Batch generation
│   ├── training/              # Training
│   │   └── trainer.py         # Training loop
│   ├── utils/                 # Utilities
│   │   └── config.py          # YAML loader (with inheritance)
│   └── cli/                   # Command-line interface
│       └── main.py            # Entry point
├── configs/
│   ├── model/                 # Model configs
│   │   ├── base.yaml          # Shared
│   │   └── tiny.yaml          # tiny (inherits base)
│   └── train/                 # Training configs
├── scripts/                   # Utility scripts
├── tests/                     # Tests
├── docs/                      # Documentation
└── data/                      # Training data (gitignored)

```

## Installation

```bash
git clone https://github.com/Dev-SJH/VORQEN.git
cd VORQEN
pip install -e .
```

Usage

Training

```bash
python -m vorqen.cli.main train --config configs/model/tiny.yaml
```

Generation

```bash
python -m vorqen.cli.main generate \
  --config configs/model/tiny.yaml \
  --prompt "Hello"
```

Execution Environments

Environment Purpose Notes
Google Colab Training (free T4 GPU) Recommended
GitHub Codespaces Code editing + git 
Acode (Alpine) Editing on mobile PyTorch not supported

Alpine Linux uses musl libc, which is incompatible with the official
PyTorch wheels. Run training on Colab.

License

MIT License — free to use, modify, and distribute

Acknowledgements

This project was heavily inspired by Andrej Karpathy's
nanoGPT.

---

Made with ❤️ by a middle schooler who refuses to give up.

```
