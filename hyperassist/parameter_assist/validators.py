# hyperassist/parameter_assist/validators.py

def validate_params(params, required=None, types=None):
    """
    Validates input params.
    - required: list of keys that must be present
    - types: dict of key: expected_type (e.g., {"batch_size": int})

    Returns (is_valid, error_msgs)
    """
    error_msgs = []
    if required:
        for key in required:
            if key not in params:
                error_msgs.append(f"Missing required parameter: '{key}'")

    if types:
        for key, expected_type in types.items():
            if key in params and not isinstance(params[key], expected_type):
                error_msgs.append(f"Parameter '{key}' should be of type {expected_type.__name__}, got {type(params[key]).__name__}")

    return (len(error_msgs) == 0), error_msgs


def warn_unknown_params(params, known_canonicals):
    """
    Warns about params that aren't recognized/aliased in your canonical mapping.
    Returns a list of warnings.
    """
    warnings = []
    for k in params:
        if k not in known_canonicals:
            warnings.append(f"Unknown or unauthorized parameter: '{k}'")
    return warnings
