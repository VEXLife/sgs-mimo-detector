"""Split Gibbs sampler for MIMO detection.

This detector introduces an auxiliary variable ``z`` to split the likelihood
and the discrete QAM prior. Each iteration alternates:
	1) Sample ``z`` from a Gaussian conditional combining the channel
	   likelihood and a quadratic coupling to the current symbols ``x``.
	2) Sample each entry of ``x`` from the discrete constellation conditioned on
	   ``z`` with weights proportional to ``exp(-|z_i - s|^2 / eta^2)``.

The coupling variance ``eta^2`` is annealed from ``eta_max`` to ``eta_min``
over ``timesteps`` iterations, tightening the link between ``z`` and the
discrete symbols as the sampler progresses.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .constellation import QAMConstellation


@dataclass
class SplitGibbsOptions:
	num_starts: int = 32
	timesteps: int = 50
	eta_min: float = 0.01
	eta_max: float = 1.0


def _sample_z(
	x: np.ndarray,
	S: np.ndarray,
	V: np.ndarray,
	Vh: np.ndarray,
	b_y: np.ndarray,
	noise_var: float,
	eta_sq: float,
) -> np.ndarray:
	"""Sample auxiliary variable z | x, y using precomputed complex SVD.

	Uses the identity with H = U Σ V^H:
	  A(η) = H^H H + α I, α = σ² / η²
	  A(η)^{-1} = V diag(λ) V^H, where λ_i = 1 / (σ_i^2 + α)
	  mean = A(η)^{-1}( H^H y + α x )
	  noise = V diag(√λ) zeta, with zeta ~ CN(0, I)
	"""

	batch, n_tx = x.shape

	# α depends on η
	alpha = noise_var / eta_sq

	# λ on transmit subspace; extend to length n_tx for possible null space
	lambda_vals = 1.0 / (S**2 + alpha)
	sqrt_lambda = np.sqrt(lambda_vals)

	# Compute b_x = V^T x for each batch sample
	# Vh: (r, n_tx), x: (batch, n_tx) -> b_x: (batch, n_tx)
	b_x = (Vh @ x.T).T

	# mean = V [ λ ⊙ (b_y + α b_x) ]
	# c_y shared across batch, c_x batch-dependent
	c_y = lambda_vals * b_y  # (n_tx,)
	c_x = (lambda_vals[np.newaxis, :] * (alpha * b_x))  # (batch, n_tx)
	inner = c_x + c_y[np.newaxis, :]  # (batch, n_tx)
	mean = (V @ inner.T).T  # (batch, n_tx)

	# Sample complex noise: zeta ~ N(0, I)
	zeta = np.random.randn(batch, n_tx).astype(np.float32)
	zeta_scaled = zeta * (np.sqrt(noise_var) * sqrt_lambda[np.newaxis, :])
	noise = (V @ zeta_scaled.T).T

	# Keep noise calculation available; return mean by default
	return mean + noise


def _sample_x_from_prior(
	z: np.ndarray,
	constellation_real: np.ndarray,
	eta_sq: float,
) -> tuple[np.ndarray, np.ndarray]:
	"""Sample x | z from discrete prior independently per symbol.

	Returns both the sampled real representation and the symbol indices.
	"""

	batch, dim = z.shape
	n_tx = dim // 2

	z_re = z[:, :n_tx]
	z_im = z[:, n_tx:]

	s_re = constellation_real[:, 0]
	s_im = constellation_real[:, 1]

	# Compute distances to each constellation point
	diff_re = z_re[:, :, np.newaxis] - s_re[np.newaxis, np.newaxis, :]
	diff_im = z_im[:, :, np.newaxis] - s_im[np.newaxis, np.newaxis, :]
	dist_sq = diff_re**2 + diff_im**2  # (batch, n_tx, M)

	# Weights proportional to exp(-|z - s|^2 / eta^2)
	log_weights = -dist_sq / eta_sq
	log_weights = log_weights - log_weights.max(axis=-1, keepdims=True)  # stability
	weights = np.exp(log_weights)
	probs = weights / weights.sum(axis=-1, keepdims=True)

	# Sample indices per symbol
	probs_flat = probs.reshape(batch * n_tx, -1)
	cdf_flat = np.cumsum(probs_flat, axis=1)
	u_flat = np.random.rand(batch * n_tx)
	# Choose first index where CDF >= u (vectorized)
	samples = np.argmax(cdf_flat >= u_flat[:, None], axis=1)
	indices = samples.reshape(batch, n_tx)

	x_re = s_re[indices]
	x_im = s_im[indices]
	x_new = np.concatenate([x_re, x_im], axis=-1)

	return x_new.astype(np.float32), indices


def split_gibbs_detect(
	channel: np.ndarray,
	y: np.ndarray,
	noise_var: float,
	constellation: QAMConstellation,
	options: SplitGibbsOptions | None = None,
) -> np.ndarray:
	"""Split Gibbs detector (Algorithm 1 of the accompanying paper)."""

	opts = options or SplitGibbsOptions()
	n_tx = channel.shape[1]

	# Real-augmented channel and observation
	h_real = np.block([[channel.real, -channel.imag], [channel.imag, channel.real]]).astype(np.float32)
	y_real = np.concatenate([y.real, y.imag]).astype(np.float32)

	constellation_real = np.stack([constellation.symbols.real, constellation.symbols.imag], axis=-1).astype(np.float32)
	
	# Precompute real SVD H = U Σ V^T
	U, S, Vh = np.linalg.svd(h_real, full_matrices=True)
	V = Vh.T
	b_y = Vh @ h_real.T @ y_real

	# Initialize x by sampling uniformly from the constellation
	M = constellation_real.shape[0]
	rng = np.random.default_rng()
	init_indices = rng.integers(0, M, size=(opts.num_starts, n_tx))
	s_re = constellation_real[:, 0]
	s_im = constellation_real[:, 1]
	x_re0 = s_re[init_indices]
	x_im0 = s_im[init_indices]
	x = np.concatenate([x_re0, x_im0], axis=-1).astype(np.float32)
	x_indices = init_indices

	# Linear schedule for eta^2 from max to min
	etas = np.linspace(opts.eta_max, opts.eta_min, opts.timesteps).astype(np.float32)

	for eta in etas:
		eta_sq = float(eta ** 2)
		z = _sample_z(x, S, V, Vh, b_y, noise_var, eta_sq)
		x, x_indices = _sample_x_from_prior(z, constellation_real, eta_sq)

	# Map sampled indices to symbols and pick best chain
	demod_symbols = constellation.map_indices_to_symbols(x_indices)

	y_preds = channel @ demod_symbols.T
	errors = np.sum(np.abs(y[:, None] - y_preds) ** 2, axis=0)
	best_idx = int(np.argmin(errors))

	return x_indices[best_idx]
