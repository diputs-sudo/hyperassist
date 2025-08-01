# Attention Dropout Recommendation

## Formula

```
attention_dropout = 0.1 (transformers), else 0.0
```

- **`model_type`**: The model architecture name (e.g., "transformer", "t5", "gpt", "bert").
- **`risk_level`**: String to indicate risk of overfitting ("normal", "high"). If set to "high", slightly increase dropout.

---

## Explanation

This rule sets **attention dropout** for models using multi-head self-attention (e.g., Transformers):

- **For transformer-based models:** Use an attention dropout rate of `0.1` (default) to regularize the attention weights and prevent overfitting.
- **If overfitting risk is high:** Use a higher value such as `0.15`.
- **For other architectures:** Use `0.0` unless there is an explicit need for regularization in attention-like layers.

---

## Origin

- **Standard Practice:** The original Transformer ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)), BERT, GPT, and T5 all use `0.1` for attention dropout.
- Increasing this to `0.15` or more is sometimes recommended for small datasets or high overfitting risk.

---

## Example Calculation

Suppose:
- `model_type = "transformer"`, `risk_level = "normal"` → `attention_dropout = 0.1`
- `model_type = "bert"`, `risk_level = "high"` → `attention_dropout = 0.15`
- `model_type = "resnet"` → `attention_dropout = 0.0`

---

## Recommended Ranges

- **Transformer models:** `0.1` (default), increase to `0.15` if overfitting.
- **Other models:** `0.0` unless you have custom attention layers that benefit from regularization.
- Typical range: **0.0 – 0.2**

---

## Python Code Example

```python
def recommend_attention_dropout(model_type, risk_level="normal"):
    if model_type.lower() in {"transformer", "t5", "gpt", "bert"}:
        return 0.1 if risk_level == "normal" else 0.15
    else:
        return 0.0

# Example usage:
print(recommend_attention_dropout("transformer"))             # Output: 0.1
print(recommend_attention_dropout("bert", risk_level="high")) # Output: 0.15
print(recommend_attention_dropout("resnet"))                  # Output: 0.0
```

---

Notes
- **Dropout is not always applied in attention layers** in non-transformer models.
- For very large models or datasets, you may reduce dropout (e.g., `0.05`), but `0.1` is safest.
- Attention dropout can interact with other forms of dropout or regularization; adjust based on validation loss and overfitting signs.

---
