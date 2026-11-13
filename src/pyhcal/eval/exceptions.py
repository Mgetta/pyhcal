"""Exception hierarchy for the evaluation pipeline."""


class EvaluationError(Exception):
    """Base exception for evaluation-package failures."""


class ConfigError(EvaluationError):
    """Invalid or inconsistent evaluation configuration or definition."""


class RegistryError(ConfigError):
    """Invalid registry key, duplicate registration, or unknown definition."""


class FrameError(EvaluationError):
    """Invalid input series or paired frame."""
