# Probabilistic outputs: configurable ensembles (SAR / lagged-IC / hybrid) + CRPS + metrics API

## Context & motivation

Probabilistic forecasting is a core research direction for neural weather models. Ensemble-aware evaluation via CRPS is the NWP standard for assessing predictive uncertainty and calibration.

Following @leifdenby's review feedback, this PR has been expanded to include both principled ensemble generation and proper ensemble evaluation as a coherent, reviewable unit.

---

## What was wrong before

- The only ensemble path added Gaussian jitter `x + ε` at every step — no clear ensemble definition, statistically equivalent to a single noisy trajectory
- `neural_lam.metrics` imports caused `AttributeError` on `get_metric` due to a package/module name collision
- No metric existed to objectively compare ensemble quality

---

## What this PR introduces

### Three named, config-driven ensemble modes

| Mode | Mechanism | Status |
|------|-----------|--------|
| `sar` | Per-step Gaussian noise, interior nodes only, BCs overwritten | Existing — retained as baseline |
| `lagged_ic` | IC perturbation along recent tendency vectors | **NEW** |
| `hybrid` | lagged_ic at t=0 + SAR noise at t=1…T | **NEW** |

```yaml
training:
  ensemble_mode: hybrid        # sar | lagged_ic | hybrid
  ensemble_size: 10
  perturbation_scale: 0.01     # SAR per-step σ
  ic_perturbation_scale: 0.05  # IC perturbation σ
```

### Why lagged_ic is not "just noise"

`sar` adds i.i.d. noise uniformly — no physical motivation, members are statistically identical random draws.

`lagged_ic` perturbs initial conditions along **recent temporal tendency vectors** (differences between consecutive states). Perturbations excite modes already active in the model trajectory — interpretable, structured IC spread consistent with lagged-average forecast ensembles in NWP.

`hybrid` combines both: IC spread at initialisation + dynamical noise accumulation, covering both initial-condition uncertainty (dominant at short lead times) and process uncertainty (longer lead times).

### CRPS — proper probabilistic metric

Energy-form CRPS, vectorised PyTorch, GPU-friendly.
Registered under key `"crps"` in `DEFINED_METRICS`:

```
CRPS(F, y) = E[|X − y|] − ½ · E[|X − X′|]
```

- Proper scoring rule — uniquely minimised when forecast = truth distribution
- Generalises MAE to probabilistic forecasts  
- Penalises both bias and underdispersion/overdispersion
- Reduces to MAE for deterministic single-member forecasts

```python
import neural_lam.metrics as m
fn = m.get_metric("crps")        # returns crps_ensemble
score = fn(predictions, target)  # predictions: (E, B, T, N, F)
```

### Metrics API fix

`neural_lam/metrics/__init__.py` performed a pragmatic `importlib` dynamic load to resolve a package/module name collision between the `metrics/` directory and `metrics.py` — which caused `AttributeError` and CI failures. This minimal bridge keeps `metrics.py` as the single source of truth.

```python
from neural_lam import metrics
metrics.get_metric("crps")   # ✓
metrics.crps_ensemble        # ✓
metrics.DEFINED_METRICS      # ✓
```

Happy to follow up with a rename refactor if maintainers prefer that over the dynamic loader.

---

## Files changed

| File | Change |
|------|--------|
| `config.py` | Added `ensemble_mode`, `ic_perturbation_scale` |
| `ar_model.py` | Validates mode, routes `generate_ensemble()` |
| `crps.py` | **New** — vectorised `crps_ensemble()`, energy-form |
| `metrics.py` | Registers `crps_ensemble` as `"crps"` in `DEFINED_METRICS` |
| `__init__.py` | `importlib` load, `get_metric()` wrapper, re-exports |
| `test_crps.py` | Shape, dtype, perfect ensemble → CRPS ≈ 0 |
| `test_ar_model_ensemble.py` | Shape, diversity, `get_metric` API |
| `test_ar_model_ensemble_modes.py` | **New** — lagged_ic + hybrid tests |

No changes to training pipeline, loss functions, or deterministic workflows.
Fully backward compatible.

---

## How to verify

```bash
pytest test_crps.py test_ar_model_ensemble.py test_ar_model_ensemble_modes.py -v
python -c "import neural_lam.metrics as m; print(m.get_metric('crps'))"
```

---

## Limitations

- `lagged_ic` uses tendency-aligned perturbations — not full bred vectors or singular vectors as in operational NWP (planned future work)
- Deep ensembles (multiple trained checkpoints) are out of scope — planned follow-up PR
- `importlib` bridge is pragmatic — open to rename refactor if preferred

---

## Planned follow-up

1. Benchmark script: CRPS comparison across modes on real validation data
2. Deep ensemble loader (multiple checkpoints)  
3. Bred vector / singular vector IC perturbations (NWP-grade)
4. Rename refactor to remove dynamic import (if requested)
