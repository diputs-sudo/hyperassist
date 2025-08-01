# Label Smoothing Recommendation

## Formula

```
label_smoothing = 0.1 for transformers, else 0.0
```

- **`model_type`**: Type or name of the model (e.g., "transformer", "t5", "bart", "gpt", "bert").

---

## Explanation

This rule recommends applying **label smoothing** for Transformer-based models:

- **Label smoothing** replaces hard “one-hot” targets with slightly softer targets (e.g., 0.9 for the true class, 0.1 spread among others).
- This acts as regularization, reducing overconfidence and improving generalization.
- **For transformers (e.g., BERT, T5, BART, GPT):** Use `label_smoothing = 0.1`
- **For other architectures:** Use `0.0` (no label smoothing), unless literature or experiments suggest otherwise.

---

## Origin

- **Common Practice:** Label smoothing is used in original Transformer ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)), BERT, T5, BART, and other sequence models.
- For classic CNNs and other models, it’s less commonly used by default.

---

## Example Calculation

Suppose:
- `model_type = "transformer"`  -> `label_smoothing = 0.1`
- `model_type = "resnet"`       -> `label_smoothing = 0.0`

---

## Recommended Ranges

- **Transformer models:** `0.1` (default). Can be tuned between `0.05–0.2` for stronger/softer smoothing.
- **Other models:** Usually `0.0`, unless you have a specific reason.

---

## Python Code Example

```python
def recommend_label_smoothing(model_type):
    if model_type.lower() in {"transformer", "t5", "bart", "gpt", "bert"}:
        return 0.1
    else:
        return 0.0

# Example usage:
print(recommend_label_smoothing("transformer"))  # Output: 0.1
print(recommend_label_smoothing("resnet"))       # Output: 0.0
```

---

## Notes
- Too much label smoothing (e.g., above 0.2) can slow down convergence or harm accuracy.
- If your model is underfitting, try reducing or disabling label smoothing.
- Some advanced models and loss functions (e.g., mixup, knowledge distillation) may require different label smoothing settings.

---