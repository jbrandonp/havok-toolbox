# Competitive Comparison Table

| Criterion | **havok-toolbox v0.6.0** | pykoopman | PyDMD | rhavok | deeptime |
|-----------|--------------------------|-----------|-------|--------|----------|
| **HAVOK implementation** | ★★★★★ Full (batch+stream+adaptive) | ★★ Basic | ★ None | ★★★★ Academic | ★★ EDMD only |
| **Multichannel** | ✅ mHAVOK | ❌ | ❌ | ❌ | ❌ |
| **Adaptive/Non-stationary** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **AutoML (Optuna)** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Explainability** | ✅ Forcing Attribution | ❌ | ❌ | ❌ | ❌ |
| **sklearn-compatible** | ✅ HavokEstimator | ✅ | ❌ | ❌ | ✅ |
| **GPU acceleration** | ✅ CuPy | ❌ | ❌ | ❌ | ❌ |
| **Streaming engine** | ✅ MQTT + CSV + Synthetic | ❌ | ❌ | ❌ | ❌ |
| **Edge of chaos** | ✅ LLE + CSD + Score | ❌ | ❌ | ❌ | ❌ |
| **Bootstrap CI** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Benchmark suite** | ✅ 5 datasets × 5 methods | ❌ | ❌ | ❌ | ❌ |
| **Dashboard** | ✅ Pro (comparison + what-if) | ❌ | ❌ | ❌ | ❌ |
| **Export** | ✅ CSV/JSON/.havok | ❌ | ❌ | ❌ | ❌ |
| **PyPI** | ✅ | ✅ | ✅ | ❌ | ✅ |
| **Tests** | 146 (incl. Hypothesis) | ~20 | ~10 | ~5 | ~50 |
| **Documentation** | README + ADRs + examples | Sphinx | Basic | README | Excellent |
| **License** | MIT | MIT | MIT | MIT | LGPL |
