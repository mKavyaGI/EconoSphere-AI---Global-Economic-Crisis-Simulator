import redis.asyncio as aioredis
import redis as sync_redis
from app.core.config import settings

# Async client for FastAPI WebSockets/caching
redis_client = aioredis.from_url(
    settings.redis_url,
    encoding="utf-8",
    decode_responses=True
)

# Sync client for RQ Workers
sync_redis_client = sync_redis.from_url(
    settings.redis_url
)
