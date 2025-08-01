## parameter_assist

### `parameter_assist.check(params, **kwargs)`

Analyze hyperparameters and recommend best practices using formula-based rules.

---

### How to use 

You don’t have to fill in every possible argument, just include the ones you know!

1. Import the module:

```python
from hyperassist import parameter_assist
```

2. Define your hyperparameters as a dictionary:

```python
params = {
    "learning_rate": 0.01,
    "dropout": 0.3,
    "weight_decay": 1e-5,
}
```

3. Call `parameter_assist.check` with your params and any additional settings:

```python
# Typical CNN example
parameter_assist.check(
    params,
    model_type="cnn",
    dataset_size=50000,
    ram_gb=16,
    num_gpus=2,
    epochs=20,
    per_device_batch=32,
)
```

4. For advanced/theory-based recommendations:

```python
N = 100_000_000  # Number of model parameters
n = 50000        # Training set size
train_loss = 0.13

parameter_assist.check(
    params,
    model_type="transformer",
    N=N,
    n=n,
    train_loss=train_loss,
    kl_func=lambda p: (1 - p) * N,            # KL(q||p) as a function of dropout p
    empirical_loss_func=lambda p: train_loss, # Empirical loss as a function of p
    p_bounds=(0.0, 0.6),
    delta=0.05,
    c=2.0,
    penalty_scale=1.0,
    min_penalty=1e-8,
)
```

**Full example** 

```python
from hyperassist import parameter_assist

params = {
    "learning_rate": 0.01,
    "dropout": 0.3,
    "weight_decay": 1e-5,
}

parameter_assist.check(
    params,
    model_type="cnn",
    dataset_size=50000,
    ram_gb=16,
    num_gpus=2,
    epochs=20,
    per_device_batch=32,
)
```

**That’s it!**
Just supply your parameter dictionary and whatever you know or the most relevant keywords for your case!

---

### **Required Argument**
```
| Name   | Type           | Description                                         |
|--------|----------------|-----------------------------------------------------|
| params | dict/object    | Dictionary or object with all your hyperparameters. |
```
---

### Recommended Minimum Arguments  
```
| Argument         | Why You Need It                                 | Example Value                  |
| -----------------| ----------------------------------------------- | ------------------------------ |
| params           | REQUIRED: Your hyperparameter dictionary        | { "learning_rate": 0.01, ... } |
| model_type       | Ensures rules match model architecture          | "cnn", "transformer"           |
| dataset_size     | Needed for scaling (if not using a file/folder) | 50000                          |
| ram_gb           | RAM for memory-based batch size estimation      | 16                             |
| num_gpus         | Resource-aware batch/worker scaling             | 1 or 4                         |
| per_device_batch | Controls batch size & accumulation              | 32                             |
| epochs           | For all learning schedule/step estimates        | 20                             |
```
---

### **All Arguments**
```
| Name                   | Type                 | Default      | Description                                                 |
|------------------------|----------------------|--------------|-------------------------------------------------------------|
| params                 | dict/object          | **required** | Dictionary/object of hyperparameter values.                 |
| model_type             | str                  | "generic"    | Model type: "cnn", "transformer", "generic", etc.           |
| compute                | str                  | "medium"     | Compute level: "medium", "high", "low".                     |
| datasets_file          | str                  | None         | Path to dataset file for size inference.                    |
| datasets_folder        | str                  | None         | Path to dataset folder for size inference.                  |
| dataset_size           | int                  | 10,000       | Number of training samples.                                 |
| input_shape            | str/int              | "512"        | Input sample shape (e.g. "3x32x32").                        |
| ram_gb                 | float/int            | auto         | System RAM (GB). Auto-detected if possible.                 |
| per_device_batch       | int                  | None         | Per-device batch size.                                      |
| num_gpus               | int                  | None         | Number of GPUs.                                             |
| num_devices            | int                  | None         | Number of devices.                                          |
| parameter_budget       | int                  | None         | Model parameter budget.                                     |
| hidden_size            | int                  | None         | Model hidden size.                                          |
| num_layers             | int                  | None         | Number of layers.                                           |
| buffer_factor          | float                | 1.5          | Buffer factor for RAM estimation.                           |
| max_batch              | int                  | 512          | Max batch size allowed.                                     |
| bytes_per_sample       | int                  | None         | Size of a sample in bytes.                                  |
| risk_level             | str                  | "medium"     | Target risk: "low", "medium", "high".                       |
| method                 | str                  | "xavier"     | Param init method: "xavier", "he", "kaiming_uniform".       |
| multiple_of            | int                  | 64           | Rounds sizes to a multiple (e.g., 64, 128).                 |
| warmup_pct             | float                | 0.05         | Warmup fraction for learning rate.                          |
| depth                  | int                  | 6            | Model depth (layers).                                       |
| vocab_size             | int                  | None         | Vocabulary size for NLP models.                             |
| train_loss             | float                | None         | Empirical training loss.                                    |
| theory                 | str/bool             | "on"         | Enable theory-based recommenders.                           |
| epochs                 | int                  | None         | Number of training epochs.                                  |
| total_steps            | int                  | None         | Total training steps.                                       |
| compute_factor         | float                | 1.0          | Scaling for batch/decay recommenders.                       |
| effective_batch_size   | int                  | None         | Effective batch size (gradient accumulation).               |
| fan_in                 | int                  | None         | Input fan-in for param init.                                |
| model_complexity       | float                | None         | Model complexity (dropout recommenders).                    |
| step                   | int                  | None         | Current optimizer/training step.                            |
| d_model                | int                  | 512          | Model dim for transformer LR schedule.                      |
| warmup_steps           | int                  | 4000         | Warmup steps for transformer LR schedule.                   |
| hessian_max_eig        | float                | None         | Max Hessian eigenvalue (curvature-aware decay).             |
| alpha                  | float                | 1.0          | Scaling factor for weight decay.                            |
| min_weight_decay       | float                | 1e-5         | Minimum weight decay.                                       |
| loss_entropy           | float                | None         | Loss entropy (entropy-aware init).                          |
| weight_var             | float                | None         | Desired weight variance.                                    |
| epsilon                | float                | 1e-8         | Small value for numerical stability.                        |
| fisher_trace           | float                | None         | Fisher information trace.                                   |
| ntk_max_eig            | float                | None         | Max NTK eigenvalue.                                         |
| lr_scale               | float                | 1.0          | Learning rate scaling factor.                               |
| gradient_noise_scale   | float                | None         | Gradient noise scale (GNS batch size).                      |
| target_variance        | float                | 1.0          | Target SGD variance (GNS).                                  |
| min_batch_size         | int                  | 1            | Minimum batch size (GNS).                                   |
| I_TX                   | float                | None         | Info bottleneck: I(T;X).                                    |
| I_TY                   | float                | None         | Info bottleneck: I(T;Y).                                    |
| beta                   | float                | 1.0          | IB tradeoff parameter.                                      |
| dropout_max            | float                | 0.5          | Max dropout for IB dropout.                                 |
| dropout_min            | float                | 0.0          | Min dropout for IB dropout.                                 |
| N                      | int                  | None         | Model parameter count (PAC-Bayes).                          |
| n                      | int                  | None         | Training set size (PAC-Bayes).                              |
| delta                  | float                | 0.05         | Confidence/risk (PAC-Bayes).                                |
| kl_func                | callable             | None         | Function of one argument `p` (dropout rate), for PAC-Bayes. |
| empirical_loss_func    | callable             | None         | Function of one argument `p`, for PAC-Bayes.                |
| p_bounds               | tuple(float, float)  | (0.0, 0.6)   | Dropout search range (PAC-Bayes).                           |
| c                      | float                | 2.0          | Constant in PAC-Bayes log term.                             |
| penalty_scale          | float                | 1.0          | Penalty multiplier (PAC-Bayes).                             |
| min_penalty            | float                | 1e-8         | Minimum penalty (PAC-Bayes).                                |
```

---


### **Function Arguments (PAC-Bayes & Advanced) — Important Note**

Some recommenders require callable arguments, such as `kl_func` or `empirical_loss_func`.  
**These must each be a function of a single argument `p` (dropout rate).**

See the full example test file:
[test/test.py](https://github.com/diputs-sudo/hyperassist/blob/main/test/test.py)

### Troubleshooting
- Signature error:
```
Provided function must accept exactly one required argument (p).
```
-> Use `lambda p: ...` only.

- Variable not defined:
```
name 'fixed_preds' is not defined
```
-> Make sure any variable used in your lambda is defined.

---