import os
import sys
import asyncio
from rq import Worker, Queue
from dotenv import load_dotenv

# Add app to path so we can import from app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.redis.client import sync_redis_client
from app.database.session import AsyncSessionLocal
from app.engine.simulation import SimulationEngine

load_dotenv()

def run_simulation_job(run_id: int):
    """Entry point for the RQ worker to run a simulation."""
    async def async_runner():
        async with AsyncSessionLocal() as session:
            engine = SimulationEngine(session, run_id)
            await engine.execute()
            
    # RQ runs jobs synchronously, so we spin up the event loop here
    # Windows Selector loop policy to prevent asyncpg issues if running on Windows
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(async_runner())

if __name__ == '__main__':
    queue = Queue('simulation_queue', connection=sync_redis_client)
    worker = Worker([queue], connection=sync_redis_client)
    print("Simulation Worker starting...")
    worker.work()
