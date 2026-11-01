from .mmse import mmse_full, mmse_diag
from .transparent import transparent

EQUALIZERS = {
    "transparent": transparent,
    "mmse_full": mmse_full,
    "mmse_diag": mmse_diag,
}

IS_H_EFF_NEEDED = {
    "transparent": False,
    "mmse_full": True,
    "mmse_diag": True,
}