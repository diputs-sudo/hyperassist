# Information-Theoretic Parameter Initialization

## Formula

```
init_scale = sqrt((loss_entropy + ε) / (weight_var + ε))
```

- **`loss_entropy`**: The entropy of the loss distribution, which measures output uncertainty (float, >0)
- **`weight_var`**: Desired variance for the weights (float, >0)
- **`ε` (epsilon)**: Small positive constant to avoid division by zero (default: `1e-8`)

---

## Intuitive Explanation

This formula sets the parameter initialization scale using information theory.  
A higher entropy in the loss distribution means the outputs are less certain, which leads to a larger initialization scale.  
If you want lower variance for weights, the scale decreases.  
Epsilon ensures numerical stability and prevents division by zero.

---

## Theoretical Background / Origin

Information-theoretic initialization methods are inspired by Minimum Description Length (MDL) principles.  
By connecting model parameter entropy and the desired information capacity, this method finds a scale that encodes just enough "information" to fit the data.  
See [Hinton & Van Camp, 1993](https://www.cs.toronto.edu/~hinton/absps/colt93.pdf) for the original MDL-based view of neural parameterization.

---

## Example Calculation

Suppose  
- `loss_entropy = 2.0`  
- `weight_var = 0.5`  
- `epsilon = 1e-8`  

Calculation  
- `init_scale = sqrt((2.0 + 1e-8) / (0.5 + 1e-8)) ≈ sqrt(4.0) = 2.0`

---

## Practical Guidance

To use this formula, estimate the loss entropy from your model outputs or literature for your problem type.  
Set `weight_var` to your desired initial variance, for example, values from standard initialization methods like He or Xavier.  
Epsilon is only for safety and is not tuned in practice.

---

## Python Example

```python
import math

def entropy_init_scale(loss_entropy, weight_var, epsilon=1e-8):
    scale = math.sqrt((loss_entropy + epsilon) / (weight_var + epsilon))
    return scale

# Example usage:
init_scale = entropy_init_scale(2.0, 0.5)
print(f"Recommended initialization scale: {init_scale:.4f}")  # Output: 2.0000
```

---

## Notes and Pitfalls
- Exact loss entropy is difficult to measure, so consider using approximate or literature values for your application.
- This initialization does not replace normalization layers or advanced initialization schemes, but it is useful for information-efficient models.
- If you use very small `weight_var` or very large `loss_entropy`, the resulting scale may be too large for standard models; adjust inputs accordingly.

---
