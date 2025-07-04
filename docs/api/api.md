# API Reference

This document covers the Python API for HyperAssist, including available modules, functions, arguments, and typical usage patterns.

---

## log_assist

### `log_assist.process([logfile: str = None])`

- **Description:** Analyze training logs for issues like instability, stuck accuracy, NaNs, or gradient explosions.
- **Arguments:**
  - `logfile` (str, optional): Path to a training log file. If omitted, reads from standard input (stdin).
- **Returns:** None (prints analysis to stdout)
- **Usage:**
    ```python
    log_assist.process("training.log")
    log_assist.process()  # For stdin
    ```

---

### `log_assist.live()`

- **Description:** Start capturing important log lines live from stdout during training.
- **Arguments:** None
- **Returns:** None
- **Usage:**
    ```python
    log_assist.live()
    # ... your training code here ...
    ```

---

### `log_assist.summarize_live()`

- **Description:** Analyze and summarize buffered log lines captured by `live()`.
- **Arguments:** None
- **Returns:** None (prints summary to stdout)
- **Usage:**
    ```python
    log_assist.summarize_live()
    ```
---
## parameter_assist

### `parameter_assist.check(params, **kwargs)`

- **Description:**  
  Analyze hyperparameters and recommend best practices using formula-based rules.  
  Automatically detects your system RAM and, if possible, your dataset size if you don’t specify them.

- **Arguments:**
  - `params` (dict or object): Dictionary or object with hyperparameter keys and values.
  - **Optional Keyword Arguments:**
    - `model_type` (str): e.g., "cnn", "transformer", or "generic".
    - `compute` (str): e.g., "medium", "high", "low". Default is "medium".
    - `datasets_file` (str): Path to a dataset file (json/csv/tsv/other); used to auto count dataset size.
    - `datasets_folder` (str): Path to a dataset folder; used to auto count dataset size.
    - `dataset_size` (int): Number of training samples. If not specified, tries to infer from `datasets_file` or `datasets_folder`, otherwise defaults to 10,000.
    - `input_shape` (str or int): Input shape of a single sample (e.g., "3x32x32" for images). If not specified, defaults to "512".
  - **Auto-detected:**
    - `ram_gb`: Detected using `psutil` (or defaults to 8GB if unavailable).
    - `dataset_size`: Inferred from file/folder if provided, else uses default.

- **Returns:**  
  None (prints detailed recommendations to stdout)

- **Usage:**
    ```python
    params = {
        "learning_rate": 0.01,
        "dropout": 0.3,
        "weight_decay": 1e-5,
    }
    # Minimum usage (relies on auto detect for RAM and dataset size)
    parameter_assist.check(params, model_type="cnn")
    
    # With explicit dataset file (auto counts lines or records)
    parameter_assist.check(params, datasets_file="my_data.json")
    
    # With all manual settings
    parameter_assist.check(
        params,
        model_type="transformer",
        compute="high",
        dataset_size=20000,
        input_shape="512"
    )
    ```

**Notes:**
- If `psutil` (it is installed automatically via pip) is installed, RAM is detected automatically.
- If no dataset size is given and no file/folder is specified, defaults to 10,000.
- If `input_shape` is not provided, defaults to 512.

---

## Example Workflows

### 1. Analyze a Training Log from a File

```python
from hyperassist import log_assist

# Analyze a saved training log (e.g., from PyTorch, TensorFlow, Keras, etc.)
log_assist.process("experiment.log")
```

### 2. Analyze Hyperparameters with Minimal Input

```python
from hyperassist import parameter_assist

params = {
    "learning_rate": 0.01,
    "dropout": 0.3,
    "weight_decay": 1e-5,
}
# Will auto detect RAM and use default dataset size
parameter_assist.check(params, model_type="cnn")
```

### 3. Analyze Hyperparameters with Auto Detected Dataset Size

```python
from hyperassist import parameter_assist

params = {
    "learning_rate": 0.001,
    "dropout": 0.4,
    "weight_decay": 1e-6,
}
# Automatically counts samples from the provided dataset file
parameter_assist.check(params, datasets_file="data/train.json")
```

### 4. Analyze Hyperparameters with All Manual Settings

```python
from hyperassist import parameter_assist

params = {
    "learning_rate": 0.0005,
    "dropout": 0.25,
    "weight_decay": 2e-5,
}
parameter_assist.check(
    params,
    model_type="transformer",
    compute="high",
    dataset_size=20000,
    input_shape="512"
)
```

### 5. Live Log Capture and Summary

```python
from hyperassist import log_assist

# Start capturing live logs (e.g., in a training loop)
log_assist.live()
# ... run your training script, which prints logs ...
log_assist.summarize_live()
```

---