# hyperassist/parameter_assist/formulas/l4_theoretical.py

# Welcome! If you’re seeing this, you must really want to dig deep into the world of AI hyperparameters.
# Maybe you’re a researcher, or maybe you’re just super curious. Either way, this script is probably for you.
# I just hope these formulas don’t end up lost or forgotten, because sharing this stuff is why I started this whole project.

# This script is for those who want to work with the real theory behind hyperparameters. 
# There’s nothing simplified here. You’ll have to bring your own values, your own estimates. 
# I’m not giving shortcuts or guesses. This is everything I know, laid out for you as it is. 

# From here, it’s in your hands.

# Thanks,
# diputs-sudo

import math
from typing import Callable, Tuple, Optional
from scipy.optimize import minimize_scalar
import inspect 

# 1. PAC-Bayes Optimal Dropout Rate
def recommend_pacbayes_dropout_rate_advanced(
    train_loss: float,
    N: int,
    n: int,
    delta: float,
    kl_func: Optional[Callable[[float], float]] = None,
    empirical_loss_func: Optional[Callable[[float], float]] = None,
    p_bounds: Tuple[float, float] = (0.0, 0.6),
    c: float = 2.0, # Constant inside PAC-Bayes bound (e.g. 2 or sqrt(n))
    penalty_scale: float = 1.0, # Multiplier on the sqrt penalty term
    min_penalty: float = 1e-8 # Safety to prevent sqrt of negative
) -> Tuple[float, str, str]:
    """
    Computes the dropout rate that minimizes a PAC-Bayes generalization bound.

    Args:
        train_loss: Empirical loss on training data
        N: Number of model parameters
        n: Training set size
        delta: Confidence parameter (e.g., 0.05)
        kl_func: Function computing KL(q || p) from dropout rate p
        empirical_loss_func: Function computing loss as a function of dropout rate p
        p_bounds: Search interval for dropout
        c: Constant in the PAC-Bayes log term
        penalty_scale: Scaling factor for the bound penalty
        min_penalty: Minimum penalty to prevent sqrt of negative values

    Returns:
        (best_p, formula_str, explanation_str)
    """
    assert 0 < delta < 1, "delta must be in (0, 1)"
    assert n > 0, "n must be positive"

    if kl_func is None:
        kl_func = lambda p: (1 - p) * N
    else:
        kl_func = _safely_wrap_unary(kl_func)

    if empirical_loss_func is None:
        empirical_loss_func = lambda p: train_loss
    else:
        empirical_loss_func = _safely_wrap_unary(empirical_loss_func)

    def pacbayes_bound(p: float) -> float:
        try:
            kl = kl_func(p)
            empirical = empirical_loss_func(p)
        except Exception as e:
            raise RuntimeError(f"Error evaluating kl_func or empirical_loss_func at p={p}: {e}")

        penalty_numerator = kl + math.log(c / delta)
        penalty = penalty_scale * math.sqrt(
            max(penalty_numerator, min_penalty) / (2 * n)
        )
        return empirical + penalty

    res = minimize_scalar(
        pacbayes_bound, bounds=p_bounds, method='bounded'
    )
    best_p = float(res.x)

    formula = (
        "dropout = argmin_p [empirical_loss_func(p) + penalty_scale * sqrt((KL(q||p) + ln(c/δ)) / (2n))], "
        f"over p in {p_bounds}, c={c}, penalty_scale={penalty_scale}"
    )
    explanation = (
        "Minimizes the PAC-Bayes generalization bound. 'kl_func' and 'empirical_loss_func' must be functions of a single "
        "dropout argument 'p'. This result gives the optimal dropout rate based on generalization risk minimization."
    )

    return best_p, formula, explanation

# 2. Curvature-Aware Weight Decay
def recommend_curvature_aware_weight_decay(
    hessian_max_eig: float,
    alpha: float = 1.0,  # Scaling factor
    min_weight_decay: float = 1e-5
) -> Tuple[float, str, str]:
    """
    Sets weight decay inversely proportional to Hessian's largest eigenvalue.

    Args:
        hessian_max_eig: Largest eigenvalue of Hessian (float)
        alpha: Optional scaling factor (default 1.0)
        min_weight_decay: Lower bound for weight decay (prevents zero/inf)

    Returns:
        (weight_decay, formula_str, explanation_str)

    Note: Some adaptive optimizers include learning rate in their effective decay.
    """
    wd = alpha / hessian_max_eig if hessian_max_eig > 0 else min_weight_decay
    formula = "weight_decay = alpha / λ_max(Hessian)"
    explanation = (
        "Chooses the dropout rate that minimizes the PAC-Bayes generalization bound. "
        "KL-divergence and empirical loss are user-defined functions of p. "
        "'c' and 'penalty_scale' are tunable constants. "
        "The result is the most theoretically justifiable dropout rate based on generalization risk minimization."
    )
    return wd, formula, explanation

# 3. Gradient Noise Scale-Based Batch Size
def recommend_gns_batch_size(
    gradient_noise_scale: float,
    target_variance: float = 1.0,
    min_batch_size: int = 1
) -> Tuple[int, str, str]:
    """
    Batch size from gradient noise scale theory (Smith et al., 2017).

    Args:
        gradient_noise_scale: Estimated GNS (float, > 0)
        target_variance: Desired SGD variance threshold (float, > 0)
        min_batch_size: Lower bound on batch size (default 1)

    Returns:
        (batch_size, formula_str, explanation_str)

    Note: In practice, batch size is at least 1.
    """
    assert gradient_noise_scale > 0, "GNS must be positive"
    assert target_variance > 0, "target_variance must be positive"
    batch = max(int(round(gradient_noise_scale / target_variance)), min_batch_size)
    formula = "batch_size = int(round(gradient_noise_scale / target_variance))"
    explanation = (
        "Keeps gradient noise variance at desired level. Batch size clamped to at least 1."
    )
    return batch, formula, explanation

# 4. Fisher/NTK Informed Learning Rate
def recommend_fisher_ntk_lr(
    fisher_trace: Optional[float] = None,
    ntk_max_eig: Optional[float] = None,
    lr_scale: float = 1.0,
    epsilon: float = 1e-8
) -> Tuple[float, str, str]:
    """
    Learning rate from Fisher or Neural Tangent Kernel theory.

    Args:
        fisher_trace: Trace of Fisher Information Matrix (float, optional)
        ntk_max_eig: Largest eigenvalue of NTK (float, optional)
        lr_scale: Optional scaling (default 1.0, sometimes 2.0)
        epsilon: Added to denominator for safety

    Returns:
        (learning_rate, formula_str, explanation_str)

    Note: Some works use learning_rate = 2 / λ_max.
    """
    if fisher_trace is not None and fisher_trace > 0:
        lr = lr_scale / (fisher_trace + epsilon)
        formula = f"learning_rate = lr_scale / (trace(Fisher) + ε), lr_scale={lr_scale}, ε={epsilon}"
        explanation = (
            "Natural gradient LR scaling. lr_scale and epsilon improve stability. "
            "Some literature uses lr_scale=2."
        )
        return lr, formula, explanation
    elif ntk_max_eig is not None and ntk_max_eig > 0:
        lr = lr_scale / (ntk_max_eig + epsilon)
        formula = f"learning_rate = lr_scale / (λ_max(NTK) + ε), lr_scale={lr_scale}, ε={epsilon}"
        explanation = (
            "NTK theory LR scaling. lr_scale and epsilon improve stability."
        )
        return lr, formula, explanation
    else:
        raise ValueError("Must provide positive fisher_trace or ntk_max_eig.")

# 5. Information-Theoretic Parameter Initialization
def recommend_entropy_init_scale(
    loss_entropy: float,
    weight_var: float,
    epsilon: float = 1e-8
) -> Tuple[float, str, str]:
    """
    Weight initialization scale from information theory.

    Args:
        loss_entropy: Entropy of the loss distribution (float, >0)
        weight_var: Desired variance for weights (float, >0)
        epsilon: Added for numerical safety.

    Returns:
        (init_scale, formula_str, explanation_str)
    """
    assert loss_entropy > 0, "loss_entropy must be positive"
    assert weight_var > 0, "weight_var must be positive"
    scale = math.sqrt((loss_entropy + epsilon) / (weight_var + epsilon))
    formula = "init_scale = sqrt((loss_entropy + ε) / (weight_var + ε))"
    explanation = (
        "Information-theoretic MDL-based init scale. Epsilon prevents div/zero."
    )
    return scale, formula, explanation

# 6. Information Bottleneck Attention Dropout
def recommend_ib_attention_dropout(
    I_TX: float,
    I_TY: float,
    beta: float = 1.0,
    dropout_max: float = 0.5,
    dropout_min: float = 0.0,
    epsilon: float = 1e-6
) -> Tuple[float, str, str]:
    """
    Attention dropout rate from the information bottleneck principle.

    Args:
        I_TX: Mutual information (representation, input), float >= 0
        I_TY: Mutual information (representation, label), float >= 0
        beta: IB tradeoff parameter (default 1.0)
        dropout_max: Maximum allowed dropout (default 0.5)
        dropout_min: Minimum allowed dropout (default 0.0)
        epsilon: For numerical safety in denominator.

    Returns:
        (dropout, formula_str, explanation_str)
    """
    assert beta > 0, "beta must be positive"
    ratio = I_TX / (I_TY + epsilon)
    raw_dropout = 0.5 * ratio / beta
    dropout = min(dropout_max, max(dropout_min, raw_dropout))
    formula = (
        "dropout = clamp[dropout_min, dropout_max](0.5 * (I(T;X) / (I(T;Y)+ε)) / beta), "
        f"dropout_min={dropout_min}, dropout_max={dropout_max}, β={beta}, ε={epsilon}"
    )
    explanation = (
        "Encourages information compression in attention. "
        "Clamped and numerically safe. β tunes tradeoff."
    )
    return dropout, formula, explanation

def _safely_wrap_unary(func: Callable) -> Callable:
    sig = inspect.signature(func)
    params = list(sig.parameters.values())

    required_positional = [
        p for p in params
        if p.default == inspect.Parameter.empty and
           p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]

    if len(required_positional) == 1:
        return func
    else:
        raise ValueError(
            f"Provided function must accept exactly one required argument (p). Got signature: {sig}"
        )