class NumericError(Exception):
    """Raised when a numeric error occurs, such as NaN or Inf values."""


class ConvergenceError(Exception):
    """Raised when a task exhausts its epoch budget without early stopping
    ever triggering, i.e. training was cut off before reaching a genuine
    plateau rather than because it stopped improving."""
