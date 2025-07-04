# CNN Learning Rate Formula

## Formula

```
lr = 0.1 * (batch_size / 256)
```

## Explanation

This is the **linear scaling rule** for convolutional neural networks (CNNs), first popularized in large scale image classification. The idea is:
- Start with a baseline learning rate of `0.1` for a batch size of 256.
- If you increase the batch size, scale the learning rate up proportionally.
- If you decrease the batch size, scale the learning rate down.

This keeps training stable and fast, even when you change the batch size to fit your hardware.

> **Example:**  
> - For `batch_size = 128`:  
>   `lr = 0.1 * (128 / 256) = 0.05`
> - For `batch_size = 512`:  
>   `lr = 0.1 * (512 / 256) = 0.2`

## Python Code Example

```python
def compute_cnn_lr(batch_size):
    # Linear scaling rule (Goyal et al., 2017)
    return 0.1 * (batch_size / 256)

# Example usage:
batch_size = 128
learning_rate = compute_cnn_lr(batch_size)
print(f"Recommended learning rate: {learning_rate:.4f}")  # Output: 0.05
```

---