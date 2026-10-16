"""Fixed function-only prompt and a lightweight static candidate-code check."""

from __future__ import annotations

import ast
import re

ALLOWED_IMPORTS = {
    "numpy",
    "pandas",
    "sklearn",
    "xgboost",
    "lightgbm",
    "catboost",
    "scipy",
}
FORBIDDEN_CALL_NAMES = {
    "open",
    "exec",
    "eval",
    "os",
    "subprocess",
    "socket",
    "__import__",
}

CODE_TEMPLATE = """Write only one Python function named build_model(X_train, y_train).
It must return an sklearn-compatible estimator or Pipeline that supports fit and
predict_proba. You may handle missing values, categorical encoding, and feature
engineering inside the returned pipeline. Do not write a script, tests, or
explanatory prose. Use imports only from numpy, pandas, sklearn, xgboost,
lightgbm, catboost, and scipy. The function receives pandas training-fold data.
Return just the function (optionally inside a ```python code fence).

Required signature:
def build_model(X_train, y_train):
    ...
    return model
"""


def extract_code(llm_text: str) -> str:
    """Return the first fenced Python block, or the full response if absent."""
    match = re.search(r"```python\s*(.*?)```", llm_text, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else llm_text.strip()


def _call_parts(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, ast.Attribute):
        return {node.attr} | _call_parts(node.value)
    return set()


# This is a sanity filter, not a security boundary.
def validate_code(code: str) -> tuple[bool, str]:
    """Apply a sanity filter only; this is not a security boundary."""
    try:
        tree = ast.parse(code)
    except SyntaxError as error:
        return False, f"Syntax error at line {error.lineno}: {error.msg}"

    has_builder = any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "build_model"
        for node in tree.body
    )
    if not has_builder:
        return False, "Code must define build_model(X_train, y_train)."

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                return False, "Relative imports are not allowed."
            imported_modules = [node.module or ""]
        else:
            imported_modules = []
        for module in imported_modules:
            root_module = module.split(".", 1)[0]
            if root_module not in ALLOWED_IMPORTS:
                return False, f"Import of {module!r} is not allowed."

        if isinstance(node, ast.Call):
            forbidden = _call_parts(node.func) & FORBIDDEN_CALL_NAMES
            if forbidden:
                name = sorted(forbidden)[0]
                return False, f"Call to {name!r} is not allowed."

    return True, "Code passed the sanity checks."
