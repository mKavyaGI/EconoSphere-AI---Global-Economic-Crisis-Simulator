import urllib.parse
from pydantic_settings import BaseSettings

class DatabaseSettings(BaseSettings):
    postgres_host: str = "127.0.0.1"
    postgres_port: int = 5432
    postgres_db: str = "econosphere"
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"

    @property
    def database_url(self) -> str:
        encoded_password = urllib.parse.quote_plus(self.postgres_password)
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{encoded_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )
