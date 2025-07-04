# Transformer Learning Rate Formula

## Formula

```
lr = d_model^(-0.5) * min(step^(-0.5), step / warmup_steps^1.5)
```

- `d_model` is the model embedding dimension (hidden size).
- `step` is the current training step (iteration, not epoch).
- `warmup_steps` is the number of steps over which the learning rate increases linearly.

## Explanation

This is the **inverse square root learning rate schedule** introduced in the original Transformer paper (“Attention Is All You Need”).

- The learning rate increases linearly during a warmup phase (`step / warmup_steps^1.5`), then decays proportionally to the inverse square root of the step (`step^(-0.5)`).
- Scaling by `d_model^(-0.5)` keeps training stable across different model sizes.
- This schedule is designed to help large transformer models converge faster and more reliably.

> **Example:**  
> - `d_model = 512`, `step = 4000`, `warmup_steps = 4000`  
> - `lr = 512^(-0.5) * min(4000^(-0.5), 4000 / 4000^1.5)`

## Python Code Example

```python
def compute_transformer_lr(step, d_model=512, warmup_steps=4000):
    import math
    scale = d_model ** -0.5
    return scale * min(step ** -0.5, step * (warmup_steps ** -1.5))

# Example usage:
step = 4000
d_model = 512
warmup_steps = 4000
learning_rate = compute_transformer_lr(step, d_model, warmup_steps)
print(f"Recommended learning rate: {learning_rate:.6f}")
```

---
