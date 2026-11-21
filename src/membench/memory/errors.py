class AdapterConfigurationError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class AdapterUnavailableError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class NamespaceNotEmptyError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class WriteTransportError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class WriteVisibilityTimeout(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class RetrievalTransportError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class UnsupportedCapabilityError(AdapterConfigurationError):
    """Raised when an adapter cannot satisfy its manifest requirements."""

    def __init__(self, message: str, *, missing: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.missing = missing


class PolicyDependencyError(AdapterConfigurationError):
    """Raised when a selected write policy lacks a required dependency."""


class StateInspectionError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class ResetVerificationError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
