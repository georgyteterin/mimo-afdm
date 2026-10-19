from .mmse import mmse_full, mmse_diag

EQUALIZERS = {
    "mmse_full": mmse_full,
    "mmse_diag": mmse_diag,
}