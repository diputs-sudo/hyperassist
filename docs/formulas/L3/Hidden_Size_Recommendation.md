# Hidden Size Recommendation

## Formula

```
hidden_size ≈ round(sqrt(parameter_budget / num_layers), multiple_of)
```

- **`parameter_budget`**: Total number of trainable parameters you want the model to have (integer).
- **`num_layers`**: Number of transformer (or dense) layers in the model.
- **`multiple_of`**: Rounds the result to the nearest multiple (e.g., 64 or 128; GPUs are most efficient when hidden sizes are multiples of 64).

---

## Explanation

This formula estimates the **hidden size** (model width) needed so that the total parameter count (dominated by hidden size squared times the number of layers) does not exceed a target budget:

- Solves for `hidden_size` such that `hidden_size^2 * num_layers ≈ parameter_budget`.
- Rounds the result up or down to the nearest convenient multiple for hardware (like 64, 128).
- This approach helps you scale models efficiently within compute or memory constraints.

---

## Origin

- **Heuristic/Theoretical:** Common practice in transformer model scaling. See [Kaplan et al., 2020 (“Scaling Laws for Neural Language Models”)](https://arxiv.org/abs/2001.08361) for more on parameter scaling, though this exact formula is a practical approximation.

---

## Example Calculation

Suppose:
- `parameter_budget = 50_000_000` (50 million)
- `num_layers = 12`
- `multiple_of = 64`

Steps:
- Estimate: `sqrt(50,000,000 / 12) ≈ sqrt(4,166,667) ≈ 2041`
- Round to nearest multiple of 64:  
  `2041 / 64 ≈ 31.89`  
  Rounded: `32 * 64 = 2048`

So the recommended hidden size is **2048**.

---

## Recommended Ranges

- **`multiple_of`**: Use **64** (standard), **128** (large models), or as required by hardware.
- **Hidden size**: Typically between **256** and **8192** for transformer models, depending on scale.
- **num_layers**: For BERT-base: 12; BERT-large: 24; adjust as needed.

---

## Python Code Example

```python
def recommend_hidden_size(parameter_budget, num_layers, multiple_of=64):
    import math
    h_est = math.sqrt(parameter_budget / num_layers)
    hidden_size = int(round(h_est / multiple_of) * multiple_of)
    return hidden_size

# Example usage:
hs = recommend_hidden_size(50_000_000, 12, 64)
print(f"Recommended hidden size: {hs}")  # Output: 2048
```

---

## Notes
- **This is an approximation**. Real parameter counts may vary slightly depending on architecture details (embeddings, layer norm, output heads, etc.).
- If you need a more precise parameter count, calculate it from your actual model config.
- Rounding up/down to nearest `multiple_of` helps with training efficiency on most GPUs/TPUs.

---