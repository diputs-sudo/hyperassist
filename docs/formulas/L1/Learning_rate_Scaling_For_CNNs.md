# Learning Rate Scaling Formula for CNNs

## Formula

```
lr = 0.1 * (batch_size / 256)
```
- `lr`: Recommended learning rate.
- `batch_size`: The batch size used for training.

## Explanation

This formula applies the **linear learning rate scaling rule** from Goyal et al. (2017), commonly used for CNNs:

- The baseline learning rate is **0.1** at batch size **256**.
- As you increase the batch size, the learning rate increases proportionally.
- This helps maintain stable training dynamics when using larger batches.

> **Example:**  
> - `batch_size = 512`  
> - `lr = 0.1 * (512 / 256) = 0.2`

## Python Code Example

```python
def recommend_cnn_learning_rate(batch_size: int) -> float:
    """
    Linear LR scaling rule from Goyal et al. (2017)
    """
    return 0.1 * (batch_size / 256)

# Example usage:
batch_size = 512
learning_rate = recommend_cnn_learning_rate(batch_size)
print(f"Recommended learning rate: {learning_rate}")  # Output: 0.2
```

## Notes

- This is an empirical guideline may require tuning for your dataset/model.
- Especially relevant for SGD on CNNs (e.g., ResNet, ImageNet).
- Use with caution for small batch sizes (<64) or non-SGD optimizers.

---