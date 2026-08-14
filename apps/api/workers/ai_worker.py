"""
RQ Worker process for long-running AI asynchronous forecasting jobs.
Executes deep neural sequence projections and graph topology calculations offline without blocking APIs.
"""
import os
import sys
import asyncio
from rq import Worker, Queue
from dotenv import load_dotenv

# Ensure app root is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.redis.client import sync_redis_client
from app.ai.services.orchestrator import AIDecisionService

load_dotenv()

def run_ai_forecast_job(iso3: str, model_type: str = "XGBOOST"):
    """Entry point for RQ background worker to compute and cache AI forecasts."""
    async def async_runner():
        service = AIDecisionService(model_type=model_type)
        # Force refresh to compute fresh inferences and update Redis cache
        await service.get_country_forecast(iso3, force_refresh=True)
        await service.get_risk_assessment(iso3, force_refresh=True)
        
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    asyncio.run(async_runner())

if __name__ == '__main__':
    queue = Queue('ai_queue', connection=sync_redis_client)
    worker = Worker([queue], connection=sync_redis_client)
    print("AI Forecasting & Decision Worker starting...")
    worker.work()
