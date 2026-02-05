# Technical Methodology: Aero-PINN-SAM

## 1. Introduction

This document provides a detailed technical explanation of the Physics-Informed Neural Network (PINN) approach used for airfoil self-noise prediction.

## 2. Mathematical Foundation

### 2.1 Lighthill's Aeroacoustic Analogy

Lighthill's seminal work (1952) established that aerodynamic sound sources can be modeled by reformulating the Navier-Stokes equations into an inhomogeneous wave equation:

$$\frac{\partial^2 \rho'}{\partial t^2} - c_0^2 \nabla^2 \rho' = \frac{\partial^2 T_{ij}}{\partial x_i \partial x_j}$$

where $T_{ij}$ is the Lighthill stress tensor.

### 2.2 Scaling Laws

For compact sources at high subsonic speeds, the acoustic power scales as:

$$P_{acoustic} \propto \rho_0 u^8 / c_0^5$$

This is the famous "8th power law" that our physics-informed loss seeks to capture.

### 2.3 Sound Pressure Level

SPL in decibels is computed as:

$$\text{SPL} = 20 \log_{10}\left(\frac{p_{rms}}{p_{ref}}\right)$$

where $p_{ref} = 20 \mu Pa$.

## 3. SAM Optimizer Theory

### 3.1 Sharpness-Aware Minimization

The SAM optimizer minimizes a modified loss:

$$L_{SAM}(w) = \max_{\|\epsilon\| \leq \rho} L(w + \epsilon)$$

This encourages the optimizer to find flat minima where the loss is stable under weight perturbations.

### 3.2 Two-Step Procedure

1. **Ascent Step**: $\hat{\epsilon} = \rho \nabla L(w) / \|\nabla L(w)\|$
2. **Descent Step**: $w \leftarrow w - \alpha \nabla L(w + \hat{\epsilon})$

## 4. Network Architecture

The AeroNet architecture consists of:

- **Input Layer**: 5 physical parameters
- **Hidden Layers**: 2 × 64 neurons with Tanh activation
- **Output Layer**: 1 neuron (normalized SPL)

Tanh activations provide smooth gradients necessary for physics-informed learning.

## 5. Training Details

- **Epochs**: 1000
- **Optimizer**: SAM with Adam base optimizer
- **Learning Rate**: 0.001
- **Rho (SAM)**: 0.05
- **Loss**: MSE on normalized SPL

## 6. References

1. Lighthill, M. J. (1952). On Sound Generated Aerodynamically.
2. Foret, P. et al. (2020). Sharpness-Aware Minimization.
3. Raissi, M. et al. (2019). Physics-Informed Neural Networks.
