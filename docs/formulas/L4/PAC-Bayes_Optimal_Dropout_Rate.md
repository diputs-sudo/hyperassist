# PAC-Bayes Optimal Dropout Rate

## Formula

```
dropout = argminₚ [empirical_loss_func(p) + penalty_scale * sqrt((KL(q‖p) + ln(c/δ)) / (2n))]
```

where the search is over `p` in `[p_min, p_max]`.

---

## Variables & Parameters

- **`train_loss`**: Empirical training loss (float)
- **`N`**: Number of model parameters (int)
- **`n`**: Training set size (int)
- **`delta`**: Confidence parameter for generalization risk (float, between 0 and 1)
- **`kl_func(p)`**: KL divergence between dropout and reference distribution (default: `(1-p) * N`)
- **`empirical_loss_func(p)`**: Training loss as a function of dropout rate (default: constant)
- **`p_bounds`**: Range for dropout rate search, e.g., `(0.0, 0.6)`
- **`c`**: Scaling for the log term in the PAC-Bayes bound (default: `2.0`)
- **`penalty_scale`**: Multiplier for penalty (default: `1.0`)
- **`min_penalty`**: Smallest penalty to avoid sqrt(negative), (default: `1e-8`)

---

## Intuitive Explanation

This formula finds the **dropout rate `p`** that **minimizes a PAC-Bayes generalization bound**:

- The **bound** combines *empirical loss* (how well you fit the data) and a *complexity penalty* (how far your model is from a simpler reference, plus a confidence term).
- The **penalty** increases as the model gets more complex or as your confidence (`delta`) increases.
- By minimizing the sum, you select a dropout rate that *balances fitting the data and not overfitting*—as guaranteed by PAC-Bayes theory.

---

## Theoretical Background / Origin

- **PAC-Bayes Theory**: Provides generalization guarantees by bounding test error in terms of train error and a KL-divergence penalty between learned and reference distributions ([McAllester, 1999](https://link.springer.com/article/10.1023/A:1007618624809); [Dziugaite & Roy, 2017](https://arxiv.org/abs/1703.11008)).
- The **KL term** measures how far your model (with dropout `p`) is from a reference.
- The **log(c/δ)** term sets the confidence/probability of the bound holding true.

---

## Example Calculation

Suppose:
- `train_loss = 0.20`
- `N = 5,000,000`
- `n = 10,000`
- `delta = 0.05`
- `kl_func(p) = (1-p)*N`
- `empirical_loss_func(p) = train_loss` (assume constant)
- `p_bounds = (0.0, 0.5)`
- `c = 2.0`
- `penalty_scale = 1.0`

Calculate the bound at, say, `p = 0.3`:

- `KL = (1 - 0.3) * 5,000,000 = 3,500,000`
- `log(c/δ) = ln(2 / 0.05) ≈ ln(40) ≈ 3.6889`
- `penalty = 1.0 * sqrt((3,500,000 + 3.6889) / (2 * 10,000)) ≈ sqrt(175,000.0018) ≈ 418.33`
- `bound = 0.20 + 418.33 ≈ 418.53`

Try various `p` in `[0, 0.5]`—the formula finds the one that gives the **lowest bound**.

---

## Practical Guidance

- **`train_loss`**: Use your empirical training loss.
- **`N`**: Total parameter count (can use all trainable parameters).
- **`n`**: Training set size (not batch size!).
- **`delta`**: Typically `0.01` to `0.1` (lower = more conservative).
- **`p_bounds`**: Typical dropout range `[0.0, 0.6]`.
- You can supply your own `kl_func` or `empirical_loss_func` for advanced use (e.g., model-dependent empirical loss).

---

## Python Example

```python
from scipy.optimize import minimize_scalar
import math

def pacbayes_dropout(
    train_loss, N, n, delta, c=2.0, penalty_scale=1.0, p_bounds=(0.0, 0.6), min_penalty=1e-8
):
    def kl_func(p): return (1 - p) * N
    def empirical_loss_func(p): return train_loss
    def pacbayes_bound(p):
        kl = kl_func(p)
        penalty = penalty_scale * math.sqrt(
            max(kl + math.log(c / delta), min_penalty) / (2 * n)
        )
        return empirical_loss_func(p) + penalty
    res = minimize_scalar(pacbayes_bound, bounds=p_bounds, method='bounded')
    return res.x

# Example usage:
best_p = pacbayes_dropout(0.2, 5_000_000, 10_000, 0.05)
print(f"PAC-Bayes optimal dropout: {best_p:.4f}")
```

---

## Notes & Pitfalls
- The **theoretical bound** is often very loose in practice; treat results as a guide, not a guarantee.
- The shape of `kl_func` or `empirical_loss_func` can affect optimal `p`; customize if you have better estimates.
- For small datasets or large models, the penalty can dominate, suggesting high dropout.
- This technique can be slow for large hyperparameter sweeps, as it requires numerical minimization.

---