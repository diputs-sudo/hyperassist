# Fisher/NTK Informed Learning Rate

## Formula

```
learning_rate = lr_scale / (trace(Fisher) + ε)
```
### or 
```
learning_rate = lr_scale / (λ_max(NTK) + ε)
```

- **`trace(Fisher)`**: The sum of the diagonal entries (trace) of the Fisher information matrix (float, >0).
- **`λ_max(NTK)`**: Largest eigenvalue of the Neural Tangent Kernel (NTK) matrix (float, >0).
- **`lr_scale`**: Scaling factor for the learning rate (default: `1.0`, sometimes `2.0` in literature).
- **`ε`**: Small positive number for numerical safety (default: `1e-8`).

---

## Intuitive Explanation

This formula selects the **learning rate** using insights from optimization theory:

- A **larger Fisher trace** or **NTK eigenvalue** means a *sharper* or *more sensitive* loss surface; thus, you should use a **smaller learning rate** for stability.
- Conversely, a flatter (less sensitive) model can tolerate a higher learning rate.
- The scaling factor (`lr_scale`) tunes overall speed; epsilon (`ε`) prevents division by zero.

---

## Theoretical Background / Origin

- **Fisher information** and **NTK** capture how sensitive your model’s output is to parameter changes.
- Optimal learning rates derived from theory (see [Amari, 1998](http://www.yaroslavvb.com/papers/amari-why.pdf), [Jacot et al., 2018 (NTK)](https://arxiv.org/abs/1806.07572)).
- Some papers (e.g., “Learning Rate is All You Need” [Baydin et al., 2017](https://arxiv.org/abs/1606.02228)) suggest `lr_scale = 2.0` for natural gradient methods.

---

## Example Calculation

Suppose:
- `fisher_trace = 800`
- `ntk_max_eig = 500`
- `lr_scale = 1.0`
- `epsilon = 1e-8`

**Using Fisher:**
- `learning_rate = 1.0 / (800 + 1e-8) ≈ 0.00125`

**Using NTK:**
- `learning_rate = 1.0 / (500 + 1e-8) ≈ 0.002`

If `lr_scale = 2.0`, learning rate doubles.

---

## Practical Guidance

- **Choose Fisher or NTK** depending on your model and available stats.
- If both are known, you can compare both and choose the more conservative (smaller) learning rate.
- For many practical scenarios, these metrics are estimated or approximated from mini-batch gradients.
- Always test learning rates on a validation set—these are theory-guided starting points.

---

## Python Example

```python
def fisher_ntk_lr(fisher_trace=None, ntk_max_eig=None, lr_scale=1.0, epsilon=1e-8):
    if fisher_trace is not None and fisher_trace > 0:
        return lr_scale / (fisher_trace + epsilon)
    elif ntk_max_eig is not None and ntk_max_eig > 0:
        return lr_scale / (ntk_max_eig + epsilon)
    else:
        raise ValueError("Must provide positive fisher_trace or ntk_max_eig.")

# Example usage:
lr = fisher_ntk_lr(fisher_trace=800)
print(f"Recommended learning rate: {lr:.5f}")  # Output: 0.00125
