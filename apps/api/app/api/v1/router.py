from fastapi import APIRouter
from app.api.v1.endpoints import health, countries, graph, scenarios, simulation, ai

api_router = APIRouter()
api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(countries.router, prefix="/countries", tags=["countries"])
api_router.include_router(graph.router, prefix="/graph", tags=["graph"])
api_router.include_router(scenarios.router, prefix="/scenarios", tags=["scenarios"])
api_router.include_router(simulation.router, prefix="/simulation", tags=["simulation"])
api_router.include_router(ai.router, prefix="/ai", tags=["ai"])

