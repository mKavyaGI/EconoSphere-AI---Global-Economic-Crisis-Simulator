from pydantic_settings import BaseSettings

class LoggingSettings(BaseSettings):
    log_level: str = "INFO"
