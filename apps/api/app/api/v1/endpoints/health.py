from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from app.database.session import AsyncSessionLocal
from app.neo4j.session import neo4j_conn
from app.redis.client import redis_client

router = APIRouter()

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

@router.get("")
async def health_check(db: AsyncSession = Depends(get_db)):
    status = {"status": "ok", "services": {}}
    
    # Check Postgres
    try:
        await db.execute(text("SELECT 1"))
        status["services"]["postgres"] = "ok"
    except Exception as e:
        status["services"]["postgres"] = f"error: {str(e)}"
        status["status"] = "error"

    # Check Neo4j
    try:
        if neo4j_conn.driver:
            await neo4j_conn.driver.verify_connectivity()
            status["services"]["neo4j"] = "ok"
        else:
            status["services"]["neo4j"] = "disconnected"
            status["status"] = "error"
    except Exception as e:
        status["services"]["neo4j"] = f"error: {str(e)}"
        status["status"] = "error"

    # Check Redis
    try:
        await redis_client.ping()
        status["services"]["redis"] = "ok"
    except Exception as e:
        status["services"]["redis"] = f"error: {str(e)}"
        status["status"] = "error"

    return status

@router.get("/live")
async def liveness_probe():
    return {"status": "alive"}

@router.get("/ready")
async def readiness_probe():
    return {"status": "ready"}
