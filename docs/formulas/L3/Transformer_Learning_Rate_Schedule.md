# Transformer Learning Rate Schedule

## Formula

```
lr = d_model^-0.5 * min(step^-0.5, step / warmup_steps^1.5)
```

- **`d_model`**: Model hidden dimension (e.g., 512, 768, 1024)
- **`step`**: Current training step (integer)
- **`warmup_steps`**: Number of steps for learning rate warmup (e.g., 4000)

---

## Explanation

This formula defines the original learning rate schedule for Transformer models, as introduced in [Vaswani et al., 2017 (“Attention is All You Need”)](https://arxiv.org/abs/1706.03762):

- The learning rate **increases linearly** during the warmup phase (`step / warmup_steps^1.5`), which helps the optimizer avoid instability at the start of training.
- After warmup, the rate **decays proportionally to `step^-0.5`**, stabilizing training for longer runs.
- The scaling factor `d_model^-0.5` adapts the schedule for the model’s size, so larger models use a smaller base learning rate.

---

## Origin

- **Source:** [“Attention is All You Need” (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762), Section 5.3 ("Training")

---

## Example Calculation

Suppose:
- `d_model = 512`
- `step = 8000`
- `warmup_steps = 4000`

Steps:
- `scale = 512^-0.5 ≈ 0.0442`
- `arg1 = 8000^-0.5 ≈ 0.0112`
- `arg2 = 8000 / (4000^1.5) ≈ 8000 / 253982 ≈ 0.0315`
- `min(arg1, arg2) = 0.0112`
- `lr = 0.0442 * 0.0112 ≈ 0.000494`

---

## Recommended Ranges

- **`warmup_steps`**: Usually between **2000 and 8000**, depending on dataset size and batch size.
- **`d_model`**: Set by model architecture (commonly 512–2048 for Transformers).
- **Tip:** Too few warmup steps can cause instability; too many can slow early learning.

---

## Python Code Example

```python
def transformer_lr(step, d_model=512, warmup_steps=4000):
    scale = d_model ** -0.5
    arg1 = step ** -0.5
    arg2 = step / (warmup_steps ** 1.5)
    return scale * min(arg1, arg2)

# Example usage:
lr = transformer_lr(step=8000, d_model=512, warmup_steps=4000)
print(f"Learning rate: {lr:.6f}")  # Output: 0.000494
```

--- 

## Notes
- Empirical Adjustment: You might multiply the result by an extra factor (e.g., 2 or 5) for some datasets or models.
- Framework Support: Many frameworks (PyTorch, TensorFlow, etc.) have built-in options for this schedule.
- Advanced: Some modern models tweak the exponent (e.g., `step^-0.7`), but `-0.5` is standard for the classic Transformer.

---