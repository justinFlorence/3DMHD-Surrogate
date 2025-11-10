# PINN-MHD-Orszag-Tang

A Physics-Informed Neural Network (PINN) implementation for modeling the 2D Orszag–Tang vortex benchmark in magnetohydrodynamics (MHD). This project explores custom loss functions, specialized activation functions, and Runge-Kutta time integration as part of a differentiable surrogate model for plasma physics simulations.

---

## 📌 Overview

This repository is the first stage of a larger project to create differentiable surrogates for high-fidelity MHD simulations using PINNs. The current benchmark is the 2D Orszag–Tang vortex, with custom inductive loss terms designed to preserve physical constraints such as momentum and magnetic divergence, inspired by classical MHD PDEs.

---

## 🔬 Project Goals

- Learn MHD dynamics directly from simulation data using PINNs.
- Enforce plasma physics through custom loss terms.
- Explore the effect of non-standard activation functions.
- Introduce Runge-Kutta-inspired time evolution to stabilize learning.
- Serve as a launchpad for scaling to 3D Hall MHD simulations.

---

## 🗂 Project Structure

```bash
pinn-mhd-orszag-tang/
├── configs/              # Configuration templates (WIP)
├── data/
│   ├── raw/              # Raw HDF5 benchmark files from AGATE (Zenodo)
│   └── processed/        # Preprocessed NumPy archive for training
├── notebooks/            # Experimentation and visualization
├── results/              # Saved results and plots
├── scripts/
│   └── train_pinn.py     # Entry point for training (non-SLURM)
├── src/
│   ├── data_utils.py     # Data preprocessing pipeline
│   ├── losses/           # Physics-informed loss functions
│   ├── models/           # PINN model architecture
│   ├── rk/               # Runge-Kutta time evolution modules
│   ├── eval.py           # Inference & evaluation utilities
│   └── train.py          # Main training script
├── requirements.txt      # Dependencies
├── README.md             # This file
└── LICENSE

