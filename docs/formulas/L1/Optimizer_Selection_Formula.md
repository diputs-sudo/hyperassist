# Optimizer Selection Formula

## Formula

```
optimizer = AdamW # for transformer/bert/gpt models
optimizer = SGD # for cnn/resnet models
optimizer = Adam # for all other models
```
- `optimizer`: The recommended optimizer algorithm.
- `model_type`: The string name of your model architecture.

## Explanation

This heuristic selects the **default optimizer** based on the type of model architecture:

- **Transformer-like models** (`"transformer"`, `"bert"`, `"gpt"`):  
  Use **AdamW** — it is standard for modern language models and large scale transformers.
- **CNNs or ResNet-like models** (`"cnn"`, `"resnet"`):  
  Use **SGD** — classical choice for convolutional architectures, especially in vision.
- **Other models**:  
  Use **Adam** — a safe default for a wide variety of model types.

> **Example:**  
> - `model_type = "bert"`  
> - `optimizer = "AdamW"`

## Python Code Example

```python
def recommend_optimizer(model_type: str) -> str:
    """
    Chooses default optimizer by model type.
    """
    mt = model_type.lower()
    if mt in {"transformer", "bert", "gpt"}:
        return "AdamW"
    elif mt in {"cnn", "resnet"}:
        return "SGD"
    else:
        return "Adam"

# Example usage:
optimizer = recommend_optimizer("ResNet")
print(f"Recommended optimizer: {optimizer}")  # Output: SGD
```

## Notes
- These are empirical defaults some architectures may benefit from alternate optimizers.
- AdamW is almost always preferable to plain Adam for transformer-style models.
- SGD is often paired with momentum for CNNs (e.g., momentum=0.9).
- For custom or hybrid architectures, use prior research or experiment.

---