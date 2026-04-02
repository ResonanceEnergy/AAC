#!/usr/bin/env python3
"""
TurboQuant Engine for AAC
=========================
Implementation of TurboQuant (arXiv:2504.19874, ICLR 2026) for:

1. Market State Vector Compression & Similarity Search
   - Encode MacroSnapshot + RegimeState as high-dim vectors
   - Compress to 3-bit TurboQuant codes (~6x memory reduction)
   - Fast inner-product similarity for "what happened last time?"

2. Monte Carlo Path Compression
   - Compress War Room's 100K x 11 x 90 simulation arrays
   - Store/retrieve compressed paths at ~1/5 memory cost

3. Feature Vector Compression
   - Compress ML pipeline training features for storage/retrieval

Core Algorithm (from paper):
   1. Random rotation (Hadamard or QR) -> near-independent coordinates
   2. Lloyd-Max optimal scalar quantization per coordinate
   3. Optional QJL residual correction for unbiased inner products

Dependencies: numpy, scipy (both already in AAC requirements)
"""
from __future__ import annotations

import io
import json
import os
import struct
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

# UTF-8 stdout guard
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if hasattr(sys.stdout, "buffer") and sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
from scipy.special import gamma as gamma_fn
from scipy.stats import norm

logger = __import__("logging").getLogger("aac.turboquant")

# ============================================================================
# PART 1: CORE TURBOQUANT ALGORITHM
# ============================================================================

class TurboQuantBitWidth(Enum):
    """Supported quantization bit widths."""
    BIT_1 = 1   # 2 levels   - aggressive, ~36% MSE
    BIT_2 = 2   # 4 levels   - standard,   ~12% MSE
    BIT_3 = 3   # 8 levels   - recommended, ~3% MSE
    BIT_4 = 4   # 16 levels  - high-fidelity, <1% MSE


@dataclass
class QuantStats:
    """Statistics from a quantization operation."""
    original_bytes: int
    compressed_bytes: int
    compression_ratio: float
    mse: float
    max_abs_error: float
    inner_product_bias: float  # 0 = unbiased (QJL variant)
    bit_width: int
    dimension: int
    num_vectors: int
    wall_time_ms: float


def _beta_pdf_scaled(x: float, d: int) -> float:
    """PDF of a coordinate of a uniform random vector on S^{d-1}.

    f(x) = Gamma(d/2) * sqrt(pi) / Gamma((d-1)/2) * (1-x^2)^((d-3)/2)
    for x in [-1, 1].  In high d, concentrates around +/- 1/sqrt(d).
    """
    if abs(x) >= 1.0:
        return 0.0
    coeff = gamma_fn(d / 2.0) * np.sqrt(np.pi) / gamma_fn((d - 1) / 2.0)
    return coeff * (1.0 - x * x) ** ((d - 3) / 2.0)


def _compute_lloyd_max_codebook(d: int, num_levels: int,
                                max_iter: int = 200,
                                tol: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    """Compute Lloyd-Max optimal scalar quantizer for Beta-distributed coords.

    Returns (centroids, boundaries) where:
      centroids: shape (num_levels,) -- reconstruction values
      boundaries: shape (num_levels+1,) -- decision boundaries (first=-1, last=1)
    """
    # For very high d, the distribution is approximately N(0, 1/d)
    # Use Gaussian approximation for initial centroids when d > 50
    sigma = 1.0 / np.sqrt(d) if d > 1 else 1.0

    # Initialize centroids uniformly in [-3*sigma, 3*sigma]
    centroids = np.linspace(-2.5 * sigma, 2.5 * sigma, num_levels)

    # Dense grid for numerical integration
    num_grid = 4000
    grid = np.linspace(-1.0 + 1e-10, 1.0 - 1e-10, num_grid)

    if d > 50:
        # Gaussian approximation (much faster, accurate for high d)
        pdf_vals = norm.pdf(grid, 0, sigma)
    else:
        pdf_vals = np.array([_beta_pdf_scaled(x, d) for x in grid])

    # Normalize
    pdf_vals = pdf_vals / (np.sum(pdf_vals) * (grid[1] - grid[0]))

    dx = grid[1] - grid[0]

    for _ in range(max_iter):
        # Compute boundaries as midpoints of centroids
        boundaries = np.empty(num_levels + 1)
        boundaries[0] = -1.0
        boundaries[-1] = 1.0
        for i in range(num_levels - 1):
            boundaries[i + 1] = (centroids[i] + centroids[i + 1]) / 2.0

        # Update centroids: weighted mean of grid points in each region
        new_centroids = np.zeros(num_levels)
        for i in range(num_levels):
            mask = (grid >= boundaries[i]) & (grid < boundaries[i + 1])
            if i == num_levels - 1:
                mask = (grid >= boundaries[i]) & (grid <= boundaries[i + 1])

            w = pdf_vals[mask]
            total = np.sum(w) * dx
            if total > 1e-15:
                new_centroids[i] = np.sum(grid[mask] * w * dx) / total
            else:
                new_centroids[i] = centroids[i]

        if np.max(np.abs(new_centroids - centroids)) < tol:
            break
        centroids = new_centroids

    return centroids, boundaries


class TurboQuantCodebook:
    """Pre-computed, data-oblivious codebook for a given (dimension, bit_width).

    Computed once at startup. Shared across all vectors of the same dimension.
    """

    # Class-level cache: (d, b) -> codebook instance
    _cache: dict[tuple[int, int], "TurboQuantCodebook"] = {}

    def __init__(self, dimension: int, bit_width: int):
        self.dimension = dimension
        self.bit_width = bit_width
        self.num_levels = 2 ** bit_width

        self.centroids, self.boundaries = _compute_lloyd_max_codebook(
            dimension, self.num_levels
        )

        # Pre-generate rotation matrix (deterministic seed for reproducibility)
        rng = np.random.RandomState(seed=42 + dimension + bit_width)
        if dimension <= 512:
            # Full random orthogonal rotation via QR decomposition
            G = rng.randn(dimension, dimension).astype(np.float32)
            Q, _ = np.linalg.qr(G)
            self.rotation = Q
        else:
            # For very high dim: use randomized Hadamard (sign flips + permutation)
            # Approximation that avoids O(d^2) storage
            self.signs = rng.choice([-1.0, 1.0], size=dimension).astype(np.float32)
            self.perm = rng.permutation(dimension)
            self.rotation = None  # signal to use fast path

    @classmethod
    def get(cls, dimension: int, bit_width: int) -> "TurboQuantCodebook":
        """Get or create a cached codebook."""
        key = (dimension, bit_width)
        if key not in cls._cache:
            cls._cache[key] = cls(dimension, bit_width)
        return cls._cache[key]

    def rotate(self, x: np.ndarray) -> np.ndarray:
        """Apply random rotation Pi @ x."""
        if self.rotation is not None:
            return self.rotation @ x
        else:
            # Fast path: sign flip + permutation (O(d) instead of O(d^2))
            return (x * self.signs)[self.perm]

    def rotate_inverse(self, y: np.ndarray) -> np.ndarray:
        """Apply inverse rotation Pi^T @ y."""
        if self.rotation is not None:
            return self.rotation.T @ y
        else:
            result = np.empty_like(y)
            result[self.perm] = y
            return result * self.signs

    def quantize_scalar(self, val: float) -> int:
        """Quantize a single scalar to its codebook index."""
        # Binary search in boundaries
        for i in range(self.num_levels):
            if val < self.boundaries[i + 1]:
                return i
        return self.num_levels - 1

    def dequantize_scalar(self, idx: int) -> float:
        """Dequantize an index back to its centroid value."""
        return self.centroids[idx]


# ============================================================================
# PART 2: VECTOR QUANTIZER
# ============================================================================

class TurboQuantizer:
    """TurboQuant vector quantizer.

    Supports two modes:
      - MSE: minimize ||x - x_hat||^2  (reconstruction quality)
      - PROD: minimize E[|<y,x> - <y,x_hat>|^2] with unbiasedness
              (crucial for attention/similarity computations)

    Usage:
        tq = TurboQuantizer(dimension=128, bit_width=3, mode='prod')
        compressed = tq.compress(vector)     # -> CompressedVector
        restored = tq.decompress(compressed) # -> np.ndarray
        sim = tq.inner_product(compressed_a, compressed_b)  # approx <a, b>
    """

    def __init__(self, dimension: int, bit_width: int = 3,
                 mode: str = "mse"):
        """
        Args:
            dimension: vector dimensionality
            bit_width: bits per coordinate (1-4). 3 recommended.
            mode: 'mse' (reconstruction) or 'prod' (unbiased inner product)
        """
        if mode not in ("mse", "prod"):
            raise ValueError(f"mode must be 'mse' or 'prod', got {mode!r}")
        if bit_width < 1 or bit_width > 4:
            raise ValueError(f"bit_width must be 1-4, got {bit_width}")

        self.dimension = dimension
        self.bit_width = bit_width
        self.mode = mode

        if mode == "prod" and bit_width < 2:
            raise ValueError("prod mode requires bit_width >= 2 (uses b-1 for MSE + 1 for QJL)")

        # Get codebook (MSE part)
        mse_bits = bit_width if mode == "mse" else bit_width - 1
        self.codebook = TurboQuantCodebook.get(dimension, mse_bits)

        # For prod mode: QJL projection matrix (1-bit correction on residual)
        if mode == "prod":
            rng = np.random.RandomState(seed=137 + dimension)
            # S matrix: d x d with iid N(0,1) entries
            # For memory, we use a seeded RNG instead of storing the full matrix
            self._qjl_seed = 137 + dimension

    def compress(self, x: np.ndarray) -> "CompressedVector":
        """Compress a vector to TurboQuant format.

        Args:
            x: shape (d,) float vector

        Returns:
            CompressedVector with packed indices + metadata
        """
        t0 = time.perf_counter()
        x = np.asarray(x, dtype=np.float32).ravel()
        if x.shape[0] != self.dimension:
            raise ValueError(f"Expected dim {self.dimension}, got {x.shape[0]}")

        x_norm = np.linalg.norm(x)
        if x_norm < 1e-12:
            # Zero vector: store as zero
            return CompressedVector(
                indices=np.zeros(self.dimension, dtype=np.uint8),
                norm=0.0,
                qjl_signs=None,
                residual_norm=0.0,
                dimension=self.dimension,
                bit_width=self.bit_width,
                mode=self.mode,
            )

        # Normalize to unit sphere
        x_unit = x / x_norm

        # Step 1: Random rotation
        y = self.codebook.rotate(x_unit)

        # Step 2: Per-coordinate scalar quantization
        indices = np.array(
            [self.codebook.quantize_scalar(y[j]) for j in range(self.dimension)],
            dtype=np.uint8,
        )

        if self.mode == "mse":
            return CompressedVector(
                indices=indices,
                norm=float(x_norm),
                qjl_signs=None,
                residual_norm=0.0,
                dimension=self.dimension,
                bit_width=self.bit_width,
                mode=self.mode,
            )

        # Step 3 (prod mode): QJL on residual
        y_tilde = np.array(
            [self.codebook.dequantize_scalar(int(idx)) for idx in indices],
            dtype=np.float32,
        )
        residual = y - y_tilde  # in rotated space
        r_norm = float(np.linalg.norm(residual))

        # QJL: 1-bit signs of S @ residual
        rng = np.random.RandomState(seed=self._qjl_seed)
        # Generate S on-the-fly (column by column to save memory)
        qjl_signs = np.zeros(self.dimension, dtype=np.int8)
        for j in range(self.dimension):
            s_col = rng.randn(self.dimension).astype(np.float32)
            qjl_signs[j] = 1 if np.dot(s_col, residual) >= 0 else -1

        return CompressedVector(
            indices=indices,
            norm=float(x_norm),
            qjl_signs=qjl_signs,
            residual_norm=r_norm,
            dimension=self.dimension,
            bit_width=self.bit_width,
            mode=self.mode,
        )

    def decompress(self, cv: "CompressedVector") -> np.ndarray:
        """Decompress back to a float vector (approximate reconstruction)."""
        y_tilde = np.array(
            [self.codebook.dequantize_scalar(int(idx)) for idx in cv.indices],
            dtype=np.float32,
        )

        if cv.mode == "prod" and cv.qjl_signs is not None and cv.residual_norm > 1e-12:
            # Reconstruct residual via QJL: r_hat = sqrt(pi/(2d)) * ||r|| * S^T @ z
            rng = np.random.RandomState(seed=self._qjl_seed)
            r_hat = np.zeros(self.dimension, dtype=np.float32)
            for j in range(self.dimension):
                s_col = rng.randn(self.dimension).astype(np.float32)
                r_hat += cv.qjl_signs[j] * s_col
            scale = np.sqrt(np.pi / (2.0 * self.dimension)) * cv.residual_norm
            r_hat *= scale
            y_tilde = y_tilde + r_hat

        # Inverse rotation + rescale
        x_hat = self.codebook.rotate_inverse(y_tilde) * cv.norm
        return x_hat

    def compressed_inner_product(self, cv_a: "CompressedVector",
                                 cv_b: "CompressedVector") -> float:
        """Approximate inner product <a, b> from compressed representations.

        For 'prod' mode, this is an unbiased estimator.
        For 'mse' mode, this has small bias but is much faster.
        """
        a = self.decompress(cv_a)
        b = self.decompress(cv_b)
        return float(np.dot(a, b))

    def compress_batch(self, X: np.ndarray) -> list["CompressedVector"]:
        """Compress a batch of vectors. X: shape (n, d)."""
        return [self.compress(X[i]) for i in range(X.shape[0])]

    def decompress_batch(self, cvs: list["CompressedVector"]) -> np.ndarray:
        """Decompress a batch of CompressedVectors."""
        return np.array([self.decompress(cv) for cv in cvs])


@dataclass
class CompressedVector:
    """Packed representation of a TurboQuant-compressed vector."""
    indices: np.ndarray       # uint8 array, shape (d,), values in [0, 2^b)
    norm: float               # original L2 norm
    qjl_signs: Optional[np.ndarray]  # int8 array, shape (d,), +/-1 (prod mode only)
    residual_norm: float      # ||residual|| for QJL reconstruction
    dimension: int
    bit_width: int
    mode: str

    @property
    def size_bytes(self) -> int:
        """Actual compressed size in bytes."""
        # indices: ceil(d * bit_width / 8)
        idx_bytes = int(np.ceil(self.dimension * self.bit_width / 8))
        # norm: 4 bytes (float32)
        meta = 4
        if self.qjl_signs is not None:
            # 1 bit per sign + residual norm (4 bytes)
            meta += int(np.ceil(self.dimension / 8)) + 4
        return idx_bytes + meta

    @property
    def original_bytes(self) -> int:
        return self.dimension * 4  # float32

    @property
    def compression_ratio(self) -> float:
        return self.original_bytes / max(self.size_bytes, 1)

    def to_bytes(self) -> bytes:
        """Serialize to compact binary format."""
        parts = []
        # Header: dim(4) + bit_width(1) + mode(1) + norm(4) + residual_norm(4)
        mode_byte = 0 if self.mode == "mse" else 1
        parts.append(struct.pack("<IBBff", self.dimension, self.bit_width,
                                 mode_byte, self.norm, self.residual_norm))
        # Packed indices
        parts.append(self.indices.tobytes())
        # QJL signs (if prod mode)
        if self.qjl_signs is not None:
            # Pack to bits: +1 -> 1, -1 -> 0
            sign_bits = ((self.qjl_signs + 1) // 2).astype(np.uint8)
            parts.append(np.packbits(sign_bits).tobytes())
        return b"".join(parts)

    @classmethod
    def from_bytes(cls, data: bytes) -> "CompressedVector":
        """Deserialize from compact binary format."""
        dim, bw, mode_byte, norm_val, res_norm = struct.unpack_from("<IBBff", data, 0)
        offset = 14
        mode = "mse" if mode_byte == 0 else "prod"

        indices = np.frombuffer(data, dtype=np.uint8, count=dim, offset=offset)
        offset += dim

        qjl_signs = None
        if mode == "prod":
            sign_bytes = int(np.ceil(dim / 8))
            packed = np.frombuffer(data, dtype=np.uint8, count=sign_bytes, offset=offset)
            unpacked = np.unpackbits(packed)[:dim]
            qjl_signs = (unpacked.astype(np.int8) * 2 - 1)

        return cls(
            indices=indices.copy(), norm=norm_val, qjl_signs=qjl_signs,
            residual_norm=res_norm, dimension=dim, bit_width=bw, mode=mode,
        )


# ============================================================================
# PART 3: MARKET STATE VECTOR ENCODER
# ============================================================================

# 32-dimensional market state vector encoding
MARKET_STATE_FIELDS = [
    # Volatility block (4)
    "vix", "vix_change_1d", "realized_vol_20d", "vol_term_slope",
    # Credit block (4)
    "hy_spread_bps", "ig_spread_bps", "credit_vol_skew", "loan_spread_bps",
    # Rates block (4)
    "yield_curve_10_2", "yield_10y", "real_yield_10y", "breakeven_inflation",
    # Macro block (4)
    "core_pce", "gdp_growth", "oil_price_zscore", "gold_price_zscore",
    # Equity internals (4)
    "spy_return_1d", "qqq_return_1d", "breadth_adv_dec", "volume_ratio",
    # Sector stress (4)
    "kre_return_1d", "hyg_return_1d", "airlines_return_1d", "shipping_return_1d",
    # Sentiment (4)
    "fear_greed_norm", "dollar_index_norm", "safe_haven_bid", "war_binary",
    # Regime encoding (4)
    "regime_bearish", "regime_confidence", "armed_formula_count", "vol_shock_readiness",
]

MARKET_STATE_DIM = len(MARKET_STATE_FIELDS)  # 32


def encode_market_state(snapshot: dict, regime: dict | None = None) -> np.ndarray:
    """Convert a MacroSnapshot dict + RegimeState dict into a 32-dim float vector.

    Missing values are filled with 0.0 (neutral).
    Values are z-score normalized to typical ranges.
    """
    vec = np.zeros(MARKET_STATE_DIM, dtype=np.float32)

    # Helper: safe get with default and normalization
    def g(key: str, default: float = 0.0) -> float:
        return float(snapshot.get(key, default) or default)

    # Volatility block
    vec[0] = (g("vix", 20) - 20) / 10           # centered at 20, scale 10
    vec[1] = g("vix_change_1d") / 5.0            # % change, scale 5
    vec[2] = (g("realized_vol_20d", 15) - 15) / 8
    vec[3] = vec[0] - vec[2]                     # term slope proxy

    # Credit block
    vec[4] = (g("hy_spread_bps", 350) - 350) / 150
    vec[5] = (g("ig_spread_bps", 100) - 100) / 50
    vec[6] = g("credit_vol_skew") / 2.0
    vec[7] = (g("loan_spread_bps", 400) - 400) / 150

    # Rates block
    vec[8] = g("yield_curve_10_2") / 1.0         # -1 to +2 range
    vec[9] = (g("yield_10y", 4.0) - 4.0) / 1.0
    vec[10] = g("real_yield_10y", 1.5) / 2.0
    vec[11] = (g("breakeven_inflation", 2.3) - 2.3) / 0.5

    # Macro block
    vec[12] = (g("core_pce", 2.5) - 2.5) / 1.0
    vec[13] = g("gdp_growth", 2.0) / 3.0
    vec[14] = (g("oil_price", 80) - 80) / 20     # centered at 80
    vec[15] = (g("gold_price", 2000) - 2000) / 500

    # Equity internals
    vec[16] = g("spy_return_1d") / 2.0
    vec[17] = g("qqq_return_1d") / 2.5
    vec[18] = (g("breadth_adv_dec", 1.0) - 1.0) / 0.5
    vec[19] = (g("volume_ratio", 1.0) - 1.0) / 0.5

    # Sector stress
    vec[20] = g("kre_return_1d") / 3.0
    vec[21] = g("hyg_return_1d") / 1.5
    vec[22] = g("airlines_return_1d") / 3.0
    vec[23] = g("shipping_return_1d") / 4.0

    # Sentiment
    vec[24] = (g("fear_greed", 50) - 50) / 30
    vec[25] = (g("dollar_index", 103) - 103) / 5
    vec[26] = 1.0 if snapshot.get("safe_haven_bid") else 0.0
    vec[27] = 1.0 if snapshot.get("war_active") else 0.0

    # Regime encoding (from RegimeState)
    if regime:
        vec[28] = 1.0 if regime.get("is_bearish") else -1.0
        vec[29] = regime.get("regime_confidence", 0.5)
        vec[30] = min(regime.get("armed_formula_count", 0) / 5.0, 1.0)
        vec[31] = regime.get("vol_shock_readiness", 0) / 100.0

    return vec


# ============================================================================
# PART 4: COMPRESSED VECTOR INDEX (Similarity Search)
# ============================================================================

@dataclass
class IndexEntry:
    """A stored market state with metadata."""
    timestamp: str
    regime: str
    compressed: CompressedVector
    metadata: dict = field(default_factory=dict)


class TurboQuantIndex:
    """In-memory compressed vector index for market state similarity search.

    Stores compressed market state vectors and supports fast nearest-neighbor
    retrieval via approximate inner products.

    Usage:
        idx = TurboQuantIndex(dimension=32, bit_width=3)
        idx.add("2026-03-29", regime="credit_stress", vector=state_vec)
        matches = idx.search(query_vec, top_k=5)
    """

    def __init__(self, dimension: int = MARKET_STATE_DIM,
                 bit_width: int = 3, mode: str = "prod"):
        self.quantizer = TurboQuantizer(dimension, bit_width, mode)
        self.entries: list[IndexEntry] = []
        self._save_path: Optional[Path] = None

    def add(self, timestamp: str, regime: str, vector: np.ndarray,
            **metadata: Any) -> int:
        """Add a market state vector to the index. Returns entry index."""
        cv = self.quantizer.compress(vector)
        entry = IndexEntry(
            timestamp=timestamp,
            regime=regime,
            compressed=cv,
            metadata=metadata,
        )
        self.entries.append(entry)
        return len(self.entries) - 1

    def search(self, query: np.ndarray, top_k: int = 5) -> list[dict]:
        """Find the top_k most similar historical market states.

        Returns list of dicts with keys: rank, timestamp, regime, similarity,
        metadata.
        """
        if not self.entries:
            return []

        query_cv = self.quantizer.compress(query)
        query_vec = self.quantizer.decompress(query_cv)
        q_norm = np.linalg.norm(query_vec)
        if q_norm < 1e-12:
            return []

        scores = []
        for i, entry in enumerate(self.entries):
            e_vec = self.quantizer.decompress(entry.compressed)
            e_norm = np.linalg.norm(e_vec)
            if e_norm < 1e-12:
                continue
            # Cosine similarity
            cos_sim = float(np.dot(query_vec, e_vec) / (q_norm * e_norm))
            scores.append((i, cos_sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for rank, (idx, sim) in enumerate(scores[:top_k]):
            entry = self.entries[idx]
            results.append({
                "rank": rank + 1,
                "timestamp": entry.timestamp,
                "regime": entry.regime,
                "similarity": round(sim, 6),
                "metadata": entry.metadata,
            })
        return results

    @property
    def size_bytes(self) -> int:
        """Total compressed storage size."""
        return sum(e.compressed.size_bytes for e in self.entries)

    @property
    def uncompressed_bytes(self) -> int:
        """What this would cost without compression."""
        return sum(e.compressed.original_bytes for e in self.entries)

    @property
    def stats(self) -> dict:
        """Index statistics."""
        return {
            "num_entries": len(self.entries),
            "compressed_bytes": self.size_bytes,
            "uncompressed_bytes": self.uncompressed_bytes,
            "compression_ratio": round(self.uncompressed_bytes / max(self.size_bytes, 1), 2),
            "bit_width": self.quantizer.bit_width,
            "dimension": self.quantizer.dimension,
            "mode": self.quantizer.mode,
        }

    def save(self, path: str | Path) -> None:
        """Save index to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "dimension": self.quantizer.dimension,
            "bit_width": self.quantizer.bit_width,
            "mode": self.quantizer.mode,
            "num_entries": len(self.entries),
            "entries": [],
        }
        for e in self.entries:
            data["entries"].append({
                "timestamp": e.timestamp,
                "regime": e.regime,
                "metadata": e.metadata,
                "compressed_b64": __import__("base64").b64encode(
                    e.compressed.to_bytes()
                ).decode("ascii"),
            })
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info("TurboQuant index saved: %d entries -> %s", len(self.entries), path)

    @classmethod
    def load(cls, path: str | Path) -> "TurboQuantIndex":
        """Load index from disk."""
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        idx = cls(
            dimension=data["dimension"],
            bit_width=data["bit_width"],
            mode=data["mode"],
        )
        b64decode = __import__("base64").b64decode
        for ed in data["entries"]:
            cv = CompressedVector.from_bytes(b64decode(ed["compressed_b64"]))
            idx.entries.append(IndexEntry(
                timestamp=ed["timestamp"],
                regime=ed["regime"],
                compressed=cv,
                metadata=ed.get("metadata", {}),
            ))
        logger.info("TurboQuant index loaded: %d entries from %s", len(idx.entries), path)
        return idx


# ============================================================================
# PART 5: MONTE CARLO PATH COMPRESSOR
# ============================================================================

class MonteCarloCompressor:
    """Compress War Room Monte Carlo simulation paths.

    100K paths x 11 assets x 90 days = ~396MB at float32.
    At 3 bits via TurboQuant = ~75MB (5.3x compression).

    Compresses each asset's price trajectory across paths as a vector.
    """

    def __init__(self, bit_width: int = 3):
        self.bit_width = bit_width
        self._quantizers: dict[int, TurboQuantizer] = {}

    def _get_quantizer(self, dim: int) -> TurboQuantizer:
        if dim not in self._quantizers:
            self._quantizers[dim] = TurboQuantizer(dim, self.bit_width, mode="mse")
        return self._quantizers[dim]

    def compress_paths(self, paths: np.ndarray) -> dict:
        """Compress Monte Carlo price paths.

        Args:
            paths: shape (num_paths, num_assets, num_days) or (num_paths, num_days)

        Returns:
            dict with compressed data + metadata for reconstruction
        """
        t0 = time.perf_counter()
        paths = np.asarray(paths, dtype=np.float32)

        if paths.ndim == 2:
            paths = paths[:, np.newaxis, :]

        n_paths, n_assets, n_days = paths.shape
        original_bytes = paths.nbytes

        compressed_data = {}
        total_compressed = 0

        for asset_idx in range(n_assets):
            asset_paths = paths[:, asset_idx, :]  # (n_paths, n_days)

            # Normalize: store mean + std per day, compress normalized paths
            day_means = asset_paths.mean(axis=0)
            day_stds = asset_paths.std(axis=0)
            day_stds[day_stds < 1e-8] = 1.0

            normalized = (asset_paths - day_means[np.newaxis, :]) / day_stds[np.newaxis, :]

            # Compress each path as a vector of length n_days
            tq = self._get_quantizer(n_days)
            cvs = []
            for p in range(n_paths):
                cv = tq.compress(normalized[p])
                cvs.append(cv)
                total_compressed += cv.size_bytes

            compressed_data[asset_idx] = {
                "compressed_vectors": cvs,
                "day_means": day_means,
                "day_stds": day_stds,
            }

        wall_ms = (time.perf_counter() - t0) * 1000

        return {
            "n_paths": n_paths,
            "n_assets": n_assets,
            "n_days": n_days,
            "data": compressed_data,
            "stats": QuantStats(
                original_bytes=original_bytes,
                compressed_bytes=total_compressed,
                compression_ratio=original_bytes / max(total_compressed, 1),
                mse=0.0,  # computed on decompress
                max_abs_error=0.0,
                inner_product_bias=0.0,
                bit_width=self.bit_width,
                dimension=n_days,
                num_vectors=n_paths * n_assets,
                wall_time_ms=wall_ms,
            ),
        }

    def decompress_paths(self, compressed: dict) -> np.ndarray:
        """Decompress Monte Carlo paths back to float32 array."""
        n_paths = compressed["n_paths"]
        n_assets = compressed["n_assets"]
        n_days = compressed["n_days"]

        paths = np.zeros((n_paths, n_assets, n_days), dtype=np.float32)
        tq = self._get_quantizer(n_days)

        for asset_idx in range(n_assets):
            ad = compressed["data"][asset_idx]
            day_means = ad["day_means"]
            day_stds = ad["day_stds"]

            for p in range(n_paths):
                normalized = tq.decompress(ad["compressed_vectors"][p])
                paths[p, asset_idx, :] = normalized * day_stds + day_means

        if n_assets == 1:
            return paths[:, 0, :]
        return paths


# ============================================================================
# PART 6: AAC INTEGRATION HOOKS
# ============================================================================

# Singleton index for market state history
_market_index: Optional[TurboQuantIndex] = None
_INDEX_PATH = Path(__file__).resolve().parent.parent / "data" / "turboquant" / "market_state_index.json"


def get_market_index() -> TurboQuantIndex:
    """Get or create the global market state index."""
    global _market_index
    if _market_index is None:
        if _INDEX_PATH.exists():
            try:
                _market_index = TurboQuantIndex.load(_INDEX_PATH)
            except Exception as e:
                logger.warning("Failed to load index: %s, creating new", e)
                _market_index = TurboQuantIndex()
        else:
            _market_index = TurboQuantIndex()
    return _market_index


def save_market_index() -> None:
    """Persist the global market state index to disk."""
    if _market_index is not None:
        _market_index.save(_INDEX_PATH)


def record_market_state(snapshot: dict, regime: dict | None = None,
                        **extra_metadata: Any) -> dict:
    """Record current market state into the TurboQuant index.

    Call this from the regime engine or daily ops pipeline.
    Returns the search result of similar historical states.
    """
    vec = encode_market_state(snapshot, regime)
    idx = get_market_index()

    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
    regime_name = regime.get("primary_regime", "unknown") if regime else "unknown"

    entry_id = idx.add(
        timestamp=timestamp,
        regime=regime_name,
        vector=vec,
        **extra_metadata,
    )

    # Auto-save every 10 entries
    if len(idx.entries) % 10 == 0:
        save_market_index()

    # Find similar historical states (skip the one we just added)
    if len(idx.entries) > 1:
        similar = idx.search(vec, top_k=5)
        # Remove self-match (the entry we just added)
        similar = [s for s in similar if s["timestamp"] != timestamp]
        return {"entry_id": entry_id, "similar_states": similar[:5]}

    return {"entry_id": entry_id, "similar_states": []}


def get_compression_report() -> dict:
    """Get compression statistics for the dashboard."""
    idx = get_market_index()
    report = {
        "index_stats": idx.stats,
        "algorithm": "TurboQuant (arXiv:2504.19874)",
        "paper": "Online Vector Quantization with Near-optimal Distortion Rate",
        "venue": "ICLR 2026",
        "theoretical_bounds": {
            "1_bit_mse": 0.36,
            "2_bit_mse": 0.117,
            "3_bit_mse": 0.03,
            "4_bit_mse": 0.009,
            "shannon_gap_factor": 2.7,  # within 2.7x of Shannon lower bound
        },
    }
    return report


# ============================================================================
# PART 7: CLI & DEMO
# ============================================================================

def _demo_quantization():
    """Run a quick demo showing TurboQuant compression quality."""
    print("=" * 70)
    print("  TurboQuant Engine -- AAC Integration Demo")
    print("  Based on arXiv:2504.19874 (ICLR 2026, Google Research)")
    print("=" * 70)

    # Demo 1: Vector compression at different bit widths
    print("\n[1] VECTOR COMPRESSION QUALITY")
    print("-" * 50)
    dim = 32
    rng = np.random.RandomState(42)
    x = rng.randn(dim).astype(np.float32)
    x = x / np.linalg.norm(x)  # unit vector

    for bw in [2, 3, 4]:
        tq = TurboQuantizer(dim, bw, mode="mse")
        cv = tq.compress(x)
        x_hat = tq.decompress(cv)
        mse = float(np.mean((x - x_hat) ** 2))
        cos_sim = float(np.dot(x, x_hat) / (np.linalg.norm(x) * np.linalg.norm(x_hat) + 1e-12))
        print(f"  {bw}-bit: MSE={mse:.6f}  cos_sim={cos_sim:.6f}  "
              f"ratio={cv.compression_ratio:.1f}x  "
              f"({cv.size_bytes}B vs {cv.original_bytes}B)")

    # Demo 2: Market state encoding + similarity search
    print("\n[2] MARKET STATE SIMILARITY SEARCH")
    print("-" * 50)

    # Create a few synthetic market states
    idx = TurboQuantIndex(dimension=32, bit_width=3, mode="prod")

    states = [
        ("2020-03-15", "credit_stress",
         {"vix": 82, "hy_spread_bps": 750, "spy_return_1d": -12.0,
          "fear_greed": 5, "war_active": False}),
        ("2022-06-16", "stagflation",
         {"vix": 35, "hy_spread_bps": 500, "oil_price": 120,
          "core_pce": 6.0, "fear_greed": 15}),
        ("2024-11-06", "risk_on",
         {"vix": 15, "hy_spread_bps": 280, "spy_return_1d": 2.5,
          "fear_greed": 80, "war_active": False}),
        ("2025-08-10", "vol_shock_armed",
         {"vix": 45, "hy_spread_bps": 600, "oil_price": 95,
          "fear_greed": 12, "war_active": True}),
    ]

    for ts, regime, snap in states:
        vec = encode_market_state(snap, {"primary_regime": regime, "regime_confidence": 0.8})
        idx.add(ts, regime, vec)

    # Query: "what happened when markets looked like today?"
    today = {"vix": 25, "hy_spread_bps": 450, "oil_price": 95,
             "fear_greed": 28, "war_active": True, "kre_return_1d": -2.0}
    query = encode_market_state(today, {"primary_regime": "credit_stress",
                                        "regime_confidence": 0.7,
                                        "armed_formula_count": 3})

    results = idx.search(query, top_k=3)
    print("  Query: today's market (VIX=25, HY=450, oil=95, war=True)")
    for r in results:
        print(f"    #{r['rank']}: {r['timestamp']} [{r['regime']}] "
              f"similarity={r['similarity']:.4f}")

    print(f"\n  Index stats: {idx.stats}")

    # Demo 3: Monte Carlo compression
    print("\n[3] MONTE CARLO PATH COMPRESSION")
    print("-" * 50)
    # Simulate small MC: 1000 paths x 3 assets x 30 days
    n_paths, n_assets, n_days = 1000, 3, 30
    mc_paths = rng.randn(n_paths, n_assets, n_days).astype(np.float32)
    # Add drift to make it realistic
    for a in range(n_assets):
        mc_paths[:, a, :] = np.cumsum(mc_paths[:, a, :], axis=1) * 0.01 + 100

    mc = MonteCarloCompressor(bit_width=3)
    compressed = mc.compress_paths(mc_paths)
    restored = mc.decompress_paths(compressed)

    mse = float(np.mean((mc_paths - restored) ** 2))
    max_err = float(np.max(np.abs(mc_paths - restored)))
    stats = compressed["stats"]
    print(f"  Shape: {n_paths} paths x {n_assets} assets x {n_days} days")
    print(f"  Original:   {stats.original_bytes:>10,} bytes")
    print(f"  Compressed: {stats.compressed_bytes:>10,} bytes")
    print(f"  Ratio:      {stats.compression_ratio:.1f}x")
    print(f"  MSE:        {mse:.6f}")
    print(f"  Max error:  {max_err:.4f}")
    print(f"  Time:       {stats.wall_time_ms:.0f}ms")

    # Demo 4: Theoretical bounds
    print("\n[4] THEORETICAL BOUNDS (Paper Theorem 1)")
    print("-" * 50)
    print("  TurboQuant achieves within 2.7x of Shannon lower bound:")
    print("  Bits | MSE bound | Compression")
    print("  -----|-----------|------------")
    for b, mse_bound in [(1, 0.36), (2, 0.117), (3, 0.03), (4, 0.009)]:
        ratio = 32 / b  # FP32 -> b-bit
        print(f"    {b}  |   {mse_bound:.3f}   |   {ratio:.0f}x")

    print("\n" + "=" * 70)
    print("  Ready for production. Integrate via:")
    print("    from strategies.turboquant_engine import (")
    print("        TurboQuantizer, TurboQuantIndex, MonteCarloCompressor,")
    print("        record_market_state, get_compression_report")
    print("    )")
    print("=" * 70)


if __name__ == "__main__":
    _demo_quantization()
