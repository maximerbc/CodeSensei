"""Syntax and formatting heuristics; these do not measure repair correctness."""

import ast
import re


def extract_code_from_response(response_text: str) -> str:
    """Extract a Python/unlabelled fence, or accept syntactically valid raw Python."""
    match = re.search(
        r"```(?:python)?[ \t]*\r?\n(.*?)```",
        response_text,
        re.DOTALL | re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()
    code = response_text.strip()
    if not code:
        return ""
    try:
        ast.parse(code)
    except (SyntaxError, ValueError):
        return ""
    return code


def evaluate_code_quality(response_text: str) -> tuple[float, dict]:
    """Weight syntax validity (0.6) and extracted-code character density (0.4).

    No generated code is executed. A high score does not establish a correct,
    efficient, or safe program.
    """
    code = extract_code_from_response(response_text)
    metrics = {"syntax": "No Python code detected", "density": 0.0}
    if not code:
        return 0.0, metrics

    score = 0.0
    try:
        ast.parse(code)
        metrics["syntax"] = "Valid"
        score = 0.6
    except (SyntaxError, ValueError):
        metrics["syntax"] = "Syntax error"

    density = len(code) / len(response_text) if response_text else 0.0
    metrics["density"] = round(density, 2)
    return round(score + density * 0.4, 2), metrics
