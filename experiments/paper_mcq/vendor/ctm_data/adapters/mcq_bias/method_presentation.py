"""Shared method identity, colours and base-first presentation for evaluations."""

METHOD_COLORS = {
    "base": "#929292",
    "rmct": "#56a578",
    "bct": "#3288bd",
    "act": "#b85c38",
    "attct": "#d98c52",
    "mlpct": "#edbc83",
    "opct": "#548f91",
}


def method_key(value):
    value = str(value).lower().replace("_", "-")
    if value in {"none", "untrained", "base"} or value.startswith(("untrained base", "base model")):
        return "base"
    if value.startswith(("rate-matching", "rmct")):
        return "rmct"
    if value.startswith(("bias-augmented-consistency", "bct")):
        return "bct"
    for name in ("attct", "mlpct", "opct", "act"):
        if value.startswith(name):
            return name
    return None


def base_first(values, key=lambda value: value):
    """User's shared display order; retain order within each method family."""
    order = {name: index for index, name in enumerate(('base', 'act', 'attct', 'mlpct', 'bct', 'opct'))}
    def rank(value):
        method = method_key(key(value))
        return 100 if method == "rmct" else order.get(method, 50)
    return sorted(values, key=rank)


def method_color(value, fallback="#888888"):
    return METHOD_COLORS.get(method_key(value), fallback)
