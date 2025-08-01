# Feedforward Network (FFN) Size Recommendation

## Formula

```
ffn_size = 4 * hidden_size
```

- **`hidden_size`**: The dimension of the main model (transformer) hidden state.
- **`ffn_size`**: The inner dimension of the feedforward (FFN) sublayer in each transformer block.

---

## Explanation

This formula sets the FFN (feedforward network) size to **four times the hidden size**, which is standard in most transformer architectures:

- The FFN sublayer in each transformer block increases the dimensionality to allow for richer, non-linear transformations.
- Scaling by 4 is a convention that balances model expressiveness with efficiency; it has been found empirically effective in both NLP and vision transformers.

---

## Origin

- **Standard Practice:** Introduced in [Vaswani et al., 2017 (“Attention is All You Need”)](https://arxiv.org/abs/1706.03762), and adopted in BERT, GPT, T5, and many other models.

---

## Example Calculation

Suppose:
- `hidden_size = 1024`

Then:
- `ffn_size = 4 * 1024 = 4096`

---

## Recommended Ranges

- **`ffn_size`**: Almost always set to exactly 4× `hidden_size` for vanilla transformers.
- For some modern architectures, the multiplier may vary (e.g., **3–8×**) for efficiency or expressiveness, but 4× is the default.

---

## Python Code Example

```python
def recommend_ffn_size(hidden_size):
    return 4 * hidden_size

# Example usage:
print(recommend_ffn_size(1024))  # Output: 4096
```

---

## Notes
- **Changing this ratio** affects model capacity and compute cost; values above 4 may help for some tasks, but increase memory and FLOPs.
- Some “efficient” transformers use a smaller multiplier (e.g., 2 or 3) to save compute for edge or mobile devices.
- Make sure `ffn_size` is divisible by the hardware-friendly factor (e.g., 64) if needed for your deployment.

---