from __future__ import annotations

import numpy as np


def binary_to_gray(value: int) -> int:
    return value ^ (value >> 1)


def gray_to_binary(value: int) -> int:
    out = value
    shift = 1
    while value >> shift:
        out ^= value >> shift
        shift += 1
    return out


class QAMConstellation:
    def __init__(self, order: int = 16, normalize: bool = True) -> None:
        self.order = order
        self.normalize = normalize
        self.bits_per_symbol = int(np.log2(self.order))
        if self.bits_per_symbol % 2 != 0:
            raise ValueError("Square QAM only (bits_per_symbol must be even)")
        self.axis_bits = self.bits_per_symbol // 2
        self.symbols, self.bit_labels = self._build_constellation()
        if self.normalize:
            self._normalize()

    def _build_constellation(self) -> tuple[np.ndarray, np.ndarray]:
        sqrt_m = int(np.sqrt(self.order))
        if sqrt_m * sqrt_m != self.order:
            raise ValueError("order must be a perfect square for square QAM")

        levels = np.arange(-(sqrt_m - 1), sqrt_m, 2, dtype=np.float64)
        symbols = np.zeros(self.order, dtype=np.complex128)
        bit_labels = np.zeros((self.order, self.bits_per_symbol), dtype=np.int8)

        for i_gray in range(sqrt_m):
            i_idx = gray_to_binary(i_gray)
            i_amp = levels[i_idx]
            for q_gray in range(sqrt_m):
                q_idx = gray_to_binary(q_gray)
                q_amp = levels[q_idx]
                label = (i_gray << self.axis_bits) | q_gray
                symbols[label] = i_amp + 1j * q_amp
                bit_labels[label] = np.array(
                    list(np.binary_repr(label, width=self.bits_per_symbol)), dtype=np.int8
                )
        return symbols, bit_labels

    def _normalize(self) -> None:
        power = np.mean(np.abs(self.symbols) ** 2)
        if power <= 0:
            raise ValueError("Constellation power must be positive")
        scale = 1.0 / np.sqrt(power)
        self.symbols = self.symbols * scale

    def map_indices_to_symbols(self, indices: np.ndarray) -> np.ndarray:
        return self.symbols[indices]

    def bits_for_indices(self, indices: np.ndarray) -> np.ndarray:
        return self.bit_labels[indices]

    def nearest_indices(self, samples: np.ndarray) -> np.ndarray:
        diff = samples.reshape(-1, 1) - self.symbols.reshape(1, -1)
        return np.argmin(np.abs(diff) ** 2, axis=1)
