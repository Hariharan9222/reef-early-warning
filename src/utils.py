import os
import sys
import types
import random
import logging
import yaml
import numpy as np
from dotenv import load_dotenv

# Set cache directories to prevent permission issues
os.environ["MPLCONFIGDIR"] = os.path.abspath(".cache/matplotlib")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

# Safe numba shim to allow shap and librosa import when numba native C extension is blocked by system policies
def setup_safe_numba_shim():
    if "numba" not in sys.modules:
        m = types.ModuleType("numba")
        dummy_dec = lambda *a, **k: (lambda f: f) if a and callable(a[0]) else (lambda f: f)
        m.njit = dummy_dec
        m.jit = dummy_dec
        m.stencil = dummy_dec
        m.guvectorize = dummy_dec
        m.vectorize = dummy_dec
        m.prange = range
        m.__path__ = []
        m.typed = types.ModuleType("numba.typed")
        m.types = types.ModuleType("numba.types")
        sys.modules["numba"] = m
        sys.modules["numba.typed"] = m.typed
        sys.modules["numba.types"] = m.types

setup_safe_numba_shim()

# Load environment variables once at module import
load_dotenv()

def get_logger(name: str = "coral_guard") -> logging.Logger:
    """Return a configured logger with standard formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

def load_config(config_path: str = "config.yaml") -> dict:
    """Load yaml configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

def set_seed(seed: int = 42) -> None:
    """Fix random seeds across python, numpy, and torch for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

def get_env_variable(var_name: str, default: str = "") -> str:
    """Retrieve an environment variable safely."""
    return os.getenv(var_name, default)
