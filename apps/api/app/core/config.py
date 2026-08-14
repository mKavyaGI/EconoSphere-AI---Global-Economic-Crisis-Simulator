from app.core.settings.base import BaseAppSettings
from app.core.settings.database import DatabaseSettings
from app.core.settings.redis import RedisSettings
from app.core.settings.neo4j import Neo4jSettings
from app.core.settings.security import SecuritySettings
from app.core.settings.logging import LoggingSettings
from app.core.settings.ai import AISettings
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(
    BaseAppSettings,
    DatabaseSettings,
    RedisSettings,
    Neo4jSettings,
    SecuritySettings,
    LoggingSettings,
    AISettings
):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()

