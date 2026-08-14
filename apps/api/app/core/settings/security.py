from pydantic_settings import BaseSettings

class SecuritySettings(BaseSettings):
    jwt_secret: str = "default-secret-change-in-prod"
    jwt_algorithm: str = "HS256"
