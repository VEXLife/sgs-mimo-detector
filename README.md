# sgs-mimo-detector

Official implementation of the paper

> **An Efficient MIMO Detector Based on Split Gibbs Sampling**
> Yuxuan Wei, An-An Lu, Xiqi Gao
> 2026 IEEE 104th Vehicular Technology Conference (VTC2026-Fall)
> DOI: [10.1109/VTC2026-FALL68024.2026.11758109](https://doi.org/10.1109/VTC2026-FALL68024.2026.11758109)

Although modern AI models can reproduce papers with near-perfect accuracy, we release this repository as a reference for researchers to verify AI-generated reimplementations of our work. Questions and issues are welcome.

## Overview

Consider a flat-fading complex baseband MIMO system

$$\mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}, \quad \mathbf{n} \sim \mathcal{CN}(0, \sigma_n^2 \mathbf{I})$$

where $\mathbf{H} \in \mathbb{C}^{n_{rx} \times n_{tx}}$ is the channel matrix and the entries of
$\mathbf{x}$ are drawn from a QAM constellation $\mathcal{A}$. Instead of optimizing the MAP
objective directly, the split Gibbs sampler (SGS) introduces an auxiliary variable $\mathbf{z}$
to split the posterior into an analytically manageable Gaussian likelihood and the discrete
constellation prior, and alternates between the two conditionals:

1. **Likelihood step**

$$\mathbf{z} \sim p(\mathbf{z} \mid \mathbf{y}, \mathbf{x}) \propto \exp\left(-\frac{\|\mathbf{y} - \mathbf{H}\mathbf{z}\|^2}{\sigma_n^2} - \frac{\|\mathbf{z} - \mathbf{x}\|^2}{\eta^2}\right)$$

   A single SVD $\mathbf{H} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^H$ is computed at initialization;
   each iteration then only needs matrix-vector products on the transmit subspace.

2. **Prior step** — performed independently per symbol:

$$x_i \sim \sum_{s \in \mathcal{A}} w_s \,\delta(x_i - s), \quad w_s = \frac{\exp\left(-\frac{|z_i - s|^2}{\eta^2}\right)}{\sum_{s' \in \mathcal{A}} \exp\left(-\frac{|z_i - s'|^2}{\eta^2}\right)}$$

The coupling variance $\eta^2$ is annealed linearly from `eta_max` to `eta_min` over the
`timesteps` iterations, tightening the link between $\mathbf{z}$ and the discrete symbols as the
sampler progresses. Multiple chains run in parallel (initialized uniformly from the
constellation), and the final estimate is the chain whose $\mathbf{H}\hat{\mathbf{x}}$ best matches $\mathbf{y}$.

As discussed in the paper, the method requires only a single SVD of the channel matrix at
initialization — no Metropolis–Hastings correction step, no continuous approximation of the
prior, and no inner loops.

## Installation

```bash
pip install .
```

Requires Python ≥ 3.10 and NumPy.

## Usage

```python
import numpy as np
from sgs_mimo_detector import QAMConstellation, SplitGibbsOptions, split_gibbs_detect

rng = np.random.default_rng(0)

constellation = QAMConstellation(order=16)  # normalized square QAM, Gray mapped
n_rx, n_tx = 12, 12

# i.i.d. Rayleigh fading channel
channel = (rng.standard_normal((n_rx, n_tx)) + 1j * rng.standard_normal((n_rx, n_tx))) / np.sqrt(2.0)

# Transmit and receive
tx_indices = rng.integers(0, constellation.order, size=n_tx)
x = constellation.map_indices_to_symbols(tx_indices)

snr_db = 20.0
noise_var = 1.0 / (10 ** (snr_db / 10.0)) / n_tx
y = channel @ x + np.sqrt(noise_var / 2.0) * (
    rng.standard_normal(n_rx) + 1j * rng.standard_normal(n_rx)
)

# Detect: returns the estimated symbol indices (shape: (n_tx,))
options = SplitGibbsOptions(num_starts=128, timesteps=100, eta_max=1.0, eta_min=0.01)
rx_indices = split_gibbs_detect(channel, y, noise_var, constellation, options)
```

### Options

| Option | Default | Description | Paper setting |
| --- | --- | --- | --- |
| `num_starts` | 32 | Number of parallel chains $R$ | 128 |
| `timesteps` | 50 | Number of sampling iterations $T$ | 100 |
| `eta_max` | 1.0 | Initial coupling parameter $\eta^2$ | 1.0 |
| `eta_min` | 0.01 | Final coupling parameter $\eta^2$ | 0.01 |

## Citation

If you use this code, please cite the paper:

```bibtex
@inproceedings{wei2026sgs,
  author    = {Yuxuan Wei and An-An Lu and Xiqi Gao},
  title     = {An Efficient MIMO Detector Based on Split Gibbs Sampling},
  booktitle = {2026 IEEE 104th Vehicular Technology Conference (VTC2026-Fall)},
  year      = {2026},
  doi       = {10.1109/VTC2026-FALL68024.2026.11758109}
}
```

## License

This project is released under the [MIT License](LICENSE).
