# Vacuity-Barrier
Conditional Lower Bounds and Measured Premises for Latent World Model Planning

 # Vacuity-Barrier : Conditional Lower Bounds and Measured Premises for Latent World Model Planning

[![arXiv Paper](https://img.shields.io/badge/arXiv-Paper-B31B1B.svg)](https://arxiv.org/)
[![License: CC0-1.0](https://img.shields.io/badge/License-CC0%201.0-lightgrey.svg)](LICENSE)
[![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)

Official repository accompanying the research paper: **Vacuity-Barrier** and associated ablation experiments on the **Push-T** environment.

---

## 🧭 Abstract
This repository contains the official codebase and experiment artifacts for investigating state representations, decoupled predictive architectures, decoder-floor measurements, and conformal prediction rollout certificates.

---

## 🛠️ Installation & Setup

Ensure you have Python 3.13+ and a CUDA-compatible GPU installed. Install dependencies using:

```bash
pip install -r requirements.txt
```
Core Dependencies
gym-pusht==0.1.6

pymunk==6.11.1

torch & torchvision

scipy

numpy

🚀 Running Experiments
1. Train Encoder-Predictor Pairs
```bash
python scripts/acegp_train.py --corpus corpus/ --out acegp_runs/
```

2. Run Oracle Ceiling Evaluation
```bash
python scripts/acegp_oracle_ceiling.py --conditions gt --episodes 50 --snapshot fork
```
📜 LicenseThis project is released under the CC0-1.0 Universal license. See LICENSE for details.   

---

### 2. `requirements.txt`

```text
torch>=2.0.0
numpy>=1.22.0
scipy>=1.9.0
gymnasium==0.29.1
gym-pusht==0.1.6
pymunk==6.11.1
pandas>=2.0.0
matplotlib>=3.5.0
```
4. LICENSE (CC0-1.0)
   
CC0 1.0 Universal

Statement of Purpose

The laws of most jurisdictions throughout the world are designed to automatically copyright and related rights to original authorship and subsequent works. 

The person who associated a work with this deed has dedicated the work to the public domain by waiving all of his or her rights to the work worldwide under copyright law, including all related and neighboring rights, to the extent allowed by law.
