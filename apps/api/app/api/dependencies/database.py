from sqlalchemy.ext.asyncio import AsyncSession
from app.database.session import AsyncSessionLocal

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
