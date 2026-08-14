class DomainException(Exception):
    """Base exception for all domain-specific errors."""
    def __init__(self, message: str, code: str = "INTERNAL_ERROR", status_code: int = 500):
        self.message = message
        self.code = code
        self.status_code = status_code
        super().__init__(self.message)

class NotFoundException(DomainException):
    def __init__(self, message: str):
        super().__init__(message=message, code="RESOURCE_NOT_FOUND", status_code=404)

class ValidationException(DomainException):
    def __init__(self, message: str):
        super().__init__(message=message, code="VALIDATION_ERROR", status_code=400)
