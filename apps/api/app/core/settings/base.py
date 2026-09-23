from typing import List
from pydantic import model_validator
from pydantic_settings import BaseSettings

class BaseAppSettings(BaseSettings):
    app_name: str = "EconoSphere AI API"
    env: str = "development"
    production_safety_mode: bool = False
    allowed_origins: List[str] = ["*"]

    @model_validator(mode="after")
    def validate_production_cors(self):
        if self.env == "production" or self.production_safety_mode:
            if "*" in self.allowed_origins:
                raise ValueError("CORS wildcard ('*') is prohibited in production/safety mode.")
            if not self.allowed_origins:
                raise ValueError("ALLOWED_ORIGINS must be explicitly set in production/safety mode.")
        return self
