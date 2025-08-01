# Number of Attention Heads Recommendation

## Formula

```
num_heads divides hidden_size evenly, usually in [8, 32]
```

- **`hidden_size`**: The hidden dimension of the model (should be divisible by `num_heads`).
- **`num_heads`**: Number of parallel attention heads. Pick a value that evenly divides `hidden_size`, typically between 8 and 32.

---

## Explanation

This formula selects the **number of attention heads** for a transformer model by ensuring:

- `num_heads` divides `hidden_size` exactly (no remainder), so each head gets the same share of features.
- Typical values are **8, 12, 16, 24, or 32**; the exact value depends on your model size and hardware.
- Too few heads can limit model expressiveness; too many can reduce the capacity of each head and slow down training.

---

## Origin

- **Standard Practice:** This is based on original and follow-up transformer architectures ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)). Even division is required for multi-head self-attention math.

---

## Example Calculation

Suppose:
- `hidden_size = 1024`

Check for possible `num_heads` (between 8 and 32):

- 1024 / 8   = 128 (integer ✔)
- 1024 / 16  = 64 (integer ✔)
- 1024 / 32  = 32 (integer ✔)

All three are valid; you might choose **16** or **32** based on convention and GPU efficiency.

---

## Recommended Ranges

- **`num_heads`**: Try values between **8 and 32**. For very large models, higher values may be possible, but are uncommon.
- **Guideline:** Choose the **largest value in [8, 32]** that evenly divides `hidden_size` and fits GPU memory.

---

## Python Code Example

```python
def recommend_num_heads(hidden_size):
    for h in range(8, 33):
        if hidden_size % h == 0:
            return h
    return 8  # fallback

# Example usage:
print(recommend_num_heads(1024))  # Output: 8 (but 16 and 32 are also valid)
```

---

## Notes
- **Minimum per-head size**: Each attention head should have at least **32–64** dimensions (e.g., `hidden_size / num_heads ≥ 32`).
- **For BERT-base**: hidden_size=768, num_heads=12; for BERT-large: 1024/16 or 1024/32.
- If your model is unusually small or large, you may need to adjust head count to ensure heads are not too small.
- Some frameworks require `num_heads` to be a power of two for efficiency.

---