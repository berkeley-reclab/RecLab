"""A set of utility functions for testing."""
import numpy as np


def rmse(predictions, targets):
    """Compute the root mean squared error (RMSE) between prediction and target vectors."""
    return np.sqrt(((predictions - targets) ** 2).mean())
