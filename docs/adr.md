# Architecture Decision Records for HAVOK Toolbox
#
# Format: ADR-NNN: Title
# Status: Proposed | Accepted | Deprecated | Superseded
# Context, Decision, Consequences

## ADR-001: HAVOK as Hankel-SVD-Forcing pipeline
Status: Accepted (2026-06-27)

Context: Need to detect regime shifts in univariate time series.
Decision: Implement Brunton et al. (2017) HAVOK algorithm.
  - Hankel matrix time-delay embedding
  - Truncated SVD for eigen-time-delay coordinates
  - Linear regression residual as intermittent forcing
Consequences: Works well for nonlinear dynamics; struggles with pure amplitude changes.
  This is expected — HAVOK detects DYNAMICS changes, not amplitude changes.

## ADR-002: Two-pass architecture (batch + streaming)
Status: Accepted (2026-06-27)

Context: Users need both offline analysis (CSV files) and real-time monitoring.
Decision: Maintain two code paths.
  - `havolib/pipeline.py`: batch HAVOK for CSV/notebook analysis
  - `havolib/engine/`: streaming engine for real-time monitoring
Consequences: Some code duplication between batch and streaming.
  Acceptable trade-off — the use cases have different constraints (batch = accuracy, stream = latency).

## ADR-003: GPU acceleration via CuPy (transparent fallback)
Status: Accepted (2026-06-27)

Context: SVD is the computational bottleneck. GPU can provide 10-100x speedup.
Decision: Add `havolib/gpu.py` as a transparent abstraction layer.
  - Auto-detects CuPy availability
  - Falls back to NumPy/SciPy silently if no GPU
  - Single import point for all linear algebra
Consequences: CuPy is an optional dependency. Users without GPU get identical results.
  The GPU path is currently limited to SVD and basic LA operations.

## ADR-004: Unified config via frozen dataclasses
Status: Accepted (2026-06-27)

Context: Configuration was scattered across 5 hardcoded constants, YAML loading code, and kwargs.
Decision: Use `@dataclass(frozen=True)` with strict `__post_init__` validation.
  - `HavokParams`: core algorithm parameters
  - `PreprocessingConfig`: preprocessing pipeline
  - `PipelineConfig`: full batch pipeline
  - `EngineConfig`: streaming engine
Consequences: All config is validated at construction time, not at use time.
  Backward-compatible via legacy `load_config()` wrapper.

## ADR-005: Model serialization via gzipped JSON + base64 arrays
Status: Accepted (2026-06-27)

Context: Need to save/load pipeline results for reproducibility and sharing.
Decision: Use .havok format = gzip'd JSON with base64-encoded numpy arrays.
  - Human-readable metadata (JSON)
  - Efficient binary arrays (base64 in JSON)
  - Self-describing (version, config, timestamp)
Consequences: Not the most space-efficient format (base64 adds ~33%).
  Acceptable for typical pipeline results (KB-MB range). For TB-scale, use HDF5.

## ADR-006: Property-based testing via Hypothesis
Status: Proposed (2026-06-27)

Context: 114 example-based tests pass but don't test invariants.
Decision: Add property-based tests for:
  - Hankel matrix properties (shape, values)
  - SVD reconstruction error
  - Forcing signal zero-mean for periodic signals
  - Surrogate preserves power spectrum
Consequences: Catches edge cases that example-based tests miss.
  Adds Hypothesis as a dev dependency.
