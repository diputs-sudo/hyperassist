# HyperAssist

**Transparent Hyperparameter Guidance and Log Analysis for Deep Learning**

HyperAssist is a free, open source tool that helps you configure, debug, and understand your deep learning experiments no cloud, no paywalls, and no hidden magic numbers. 

It analyzes your training logs for common problems, recommends research-backed hyperparameters with clear formulas, and explains every suggestion so you can learn and improve your workflow.

---

## Features

- **Actionable Log Analysis:** Instantly flags unstable training, exploding gradients, and suspicious accuracy plateaus from your training logs.
- **Transparent Parameter Recommendations:** Suggests learning rate, batch size, dropout, and weight decay using formulas sourced from real research and best practices.
- **Explanations for Everything:** Every recommendation comes with a formula and reasoning. No more “try 0.001 because everyone does.”
- **Privacy First:** All analysis is fully local no data leaves your machine, no signups required.
- **Flexible API:** Use as a Python module or (coming soon) as a CLI tool.
- **Free and Open Source:** No paywalls, no quotas, no cloud dependencies.

---

## Quick Start

```bash
pip install hyperassist
```

**Analyze a training log (from a file):**
```python
from hyperassist import log_assist

log_assist.process("my_training_log.txt")
```

**Analyze hyperparameters and get recommendations:**
```python
from hyperassist import parameter_assist

params = {
    "learning_rate": 0.01,
    "per_device_train_batch_size": 16,
    "dropout": 0.5,
    "weight_decay": 1e-4
}

parameter_assist.check(
    params,
    model_type="cnn",
    dataset_size=10000,
    input_shape="3x32x32"
)
```
For more examples, see the [examples directory](https://github.com/diputs-sudo/hyperassist/docs/api/).

---

## Why HyperAssist?

Most deep learning tools only visualize metrics or perform black box tuning. HyperAssist goes further it explains *why* a value is suggested, and helps you learn good practices as you work.

All formulas and heuristics are documented and referenced so you’re never left guessing.

---

## Supported Parameters and Formulas

- **Learning Rate:**  
  - CNNs: Linear scaling rule 
  - Transformers: Inverse square root schedule
  - Generic: Dataset-size scaling rule
- **Batch Size:**  
  - RAM- and compute-aware recommendation
- **Dropout:**  
  - Adaptive to model complexity
- **Weight Decay:**  
  - Scaled for compute resources

See [docs/formulas/](https://github.com/diputs-sudo/hyperassist/docs/formulas) for full details and explanations.

---

## Documentation

- [Formulas and Explanations](./docs/formulas/)
- [API Reference](./docs/api.md)
- [FAQ & Troubleshooting](./docs/faq.md) *(coming soon)*

---

## Contributing

Contributions, suggestions, and corrections are welcome. Please open issues or pull requests, or share feedback and new formulas from research and practice.

---

## License

Apache-2.0

---

## Acknowledgments

HyperAssist is built on lessons learned from real research papers, blog posts, and the deep learning community.  
For full formula references, see [docs/formulas.md](./docs/formulas/).
