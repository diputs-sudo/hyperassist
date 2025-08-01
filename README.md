# HyperAssist v0.0.3 

**Transparent Hyperparameter Guidance and Log Analysis for Deep Learning**

HyperAssist is a free, open source tool that helps you configure, debug, and understand your deep learning experiments no cloud, no paywalls, and no hidden magic numbers. 

It analyzes your training logs for common problems, recommends research backed hyperparameters with clear formulas, and explains every suggestion so you can learn and improve your workflow.

---

## Features

- **Actionable Log Analysis:** Instantly flags unstable training, exploding gradients, and suspicious accuracy plateaus from your training logs.
- **Transparent Parameter Recommendations:** Suggests learning rate, batch size, dropout, and weight decay using formulas sourced from real research and best practices.
- **Explanations for Everything:** Every recommendation comes with a formula and reasoning. No more “try 0.001 because everyone does.”
- **Privacy First:** All analysis is fully local no data leaves your machine, no signups required.
- **Flexible API:** Use as a Python module or (coming soon) as a CLI tool.
- **Free and Open Source:** No paywalls, no quotas, no cloud dependencies.

---

## What's New in HyperAssist v0.0.3 
HyperAssist 0.0.3 is a massive leap forward, bringing 26 theory backed formulas and 57 configurable knobs, empowering you to tune like a top research lab without brute force.

### Upgrades

- **Formula Library Expanded:**
  - **6 -> 26 formulas** spanning:
    - **L1:** Core heuristics
    - **L2:** Modern scaling laws
    - **L3:** Transformer “secret sauce”
    - **L4:** Cutting-edge research (PAC-Bayes, Fisher/NTK, Information Bottleneck)
  - *These are the same theoretical levers powering today’s big models.*

- **Full L4 Support (Research-Level):**
  - PAC-Bayes optimal dropout solver
  - Curvature-aware weight decay (`λ_max(Hessian)` scaling)
  - Fisher/NTK-informed learning rate
  - Information bottleneck-driven attention dropout
  - Entropy/MDL-based parameter initialization
  - Gradient Noise Scale batch sizing

- **57 Contextual Knobs for Precision Tuning:**
  - Model architecture, dataset, compute hardware, theoretical priors, information theory signals, and more
  - Partial configs still work — **provide what you know, HyperAssist fills the rest.**

- **Blazing Fast Execution:**
  - All **26 formulas** calculated in **~1s** (even with L4 enabled)
    - *First run may take slightly longer while cache is built.*
  - No sweeps. No trial-and-error. **Pure math.**

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
For fully API explained, see the [API directory](https://github.com/diputs-sudo/hyperassist/tree/main/docs/api)
For more examples, see the [test/example file](https://github.com/diputs-sudo/hyperassist/blob/main/test/test.py).

---

## Why HyperAssist?

Most deep learning tools only visualize metrics or perform black box tuning. HyperAssist goes further it explains *why* a value is suggested, and helps you learn good practices as you work.

All formulas and heuristics are documented and referenced so you’re never left guessing.

---

## Documentation

- [Formulas and Explanations](https://github.com/diputs-sudo/hyperassist/tree/main/docs/formulas)
- [API Reference](https://github.com/diputs-sudo/hyperassist/tree/main/docs/api)
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
For full formula references, see [docs/formulas.md](https://github.com/diputs-sudo/hyperassist/tree/main/docs/formulas).

---