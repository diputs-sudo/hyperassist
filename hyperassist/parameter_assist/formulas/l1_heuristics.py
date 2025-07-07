# hyperassist/parameter_assist/formulas/l1_heristics.py 

from typing import Tuple 

def recommend_cnn_learning_rate(batch_size: int) -> Tuple[float, str, str]:
    """
    Empirical LR scaling rule for CNNs from Goyal et al. (2017)
    """
    lr = 0.1 * (batch_size / 256)
    formula = "lr = 0.1 * (batch_size /256)"
    explanation = (
        "Uses the linear scaling rule from Goyal et al. (2017), where learning rate scales"
        "propeortionally with batch size. Baseline: 0.1 at batch size 256."
    )
    return lr, formula, explanation 

