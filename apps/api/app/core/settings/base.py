from pydantic_settings import BaseSettings

class BaseAppSettings(BaseSettings):
    app_name: str = "EconoSphere AI API"
    env: str = "development"
