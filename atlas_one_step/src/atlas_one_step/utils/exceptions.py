"""Custom exceptions used by atlas_one_step."""


class AtlasError(Exception):
    """Base exception for ATLAS pipeline failures."""


class ConfigError(AtlasError):
    """Raised when config composition/validation fails."""


class DataError(AtlasError):
    """Raised for dataset path or format problems."""
