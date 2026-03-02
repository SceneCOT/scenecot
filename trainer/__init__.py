from .leo_trainer import LeoTrainer

try:
    from .leo_scaler import LeoScaler
except ModuleNotFoundError:
    LeoScaler = None
