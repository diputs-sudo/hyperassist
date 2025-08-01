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