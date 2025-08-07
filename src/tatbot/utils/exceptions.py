"""Custom exception classes for tatbot."""


class TatbotError(Exception):
    """Base exception class for all tatbot errors."""
    pass


class ConfigurationError(TatbotError):
    """Raised when there's an error in configuration."""
    pass


class NetworkConnectionError(TatbotError):
    """Raised when network/SSH connections fail."""
    pass


class HardwareError(TatbotError):
    """Raised when robot hardware operations fail."""
    pass


class SerializationError(TatbotError):
    """Raised when YAML/JSON serialization fails."""
    pass


class CalibrationError(TatbotError):
    """Raised when camera or robot calibration fails."""
    pass


class FileOperationError(TatbotError):
    """Raised when file operations fail."""
    pass