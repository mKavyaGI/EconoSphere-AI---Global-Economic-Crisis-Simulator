import json
import asyncio
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies.database import get_db
from app.schemas.simulation import SimulationStartRequest, SimulationRunResponse, SimulationSnapshotResponse
from app.services.simulation import SimulationService, ISimulationService
from app.models.simulation import LogSeverity
from app.redis.client import redis_client

router = APIRouter()


def get_simulation_service(db: AsyncSession = Depends(get_db)) -> ISimulationService:
    """Dependency injection factory for ISimulationService."""
    return SimulationService(session=db)


@router.post("/start", response_model=SimulationRunResponse, status_code=status.HTTP_201_CREATED)
async def start_simulation(
    request: SimulationStartRequest,
    service: ISimulationService = Depends(get_simulation_service)
):
    """Starts a new simulation run, validates topological dependencies, and dispatches to Redis RQ worker."""
    try:
        config_dict = request.config.model_dump() if request.config else {}
        run = await service.start_simulation(scenario_id=request.scenario_id, config_data=config_dict)
        return run
    except ValueError as err:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(err))
    except RuntimeError as err:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(err))


@router.get("", response_model=List[SimulationRunResponse])
async def list_simulations(
    scenario_id: Optional[int] = Query(None, description="Filter by scenario ID"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    service: ISimulationService = Depends(get_simulation_service)
):
    """Lists recent simulation execution runs with pagination and filtering."""
    return await service.list_simulation_runs(scenario_id=scenario_id, limit=limit, offset=offset)


@router.get("/{run_id}/status", response_model=SimulationRunResponse)
async def get_simulation_status(
    run_id: int,
    service: ISimulationService = Depends(get_simulation_service)
):
    """Polls the current status, timestamps, and configuration of an active simulation run."""
    run = await service.get_simulation_status(run_id)
    if not run:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Simulation run not found")
    return run


@router.post("/{run_id}/stop", response_model=SimulationRunResponse)
async def stop_simulation(
    run_id: int,
    service: ISimulationService = Depends(get_simulation_service)
):
    """Terminates or pauses an active simulation run."""
    run = await service.stop_simulation(run_id)
    if not run:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Simulation run not found")
    return run


@router.get("/{run_id}/snapshots", response_model=List[SimulationSnapshotResponse])
async def get_simulation_snapshots(
    run_id: int,
    service: ISimulationService = Depends(get_simulation_service)
):
    """Retrieves generated timeline propagation snapshots for visual rendering."""
    run = await service.get_simulation_status(run_id)
    if not run:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Simulation run not found")
    return await service.get_run_snapshots(run_id)


@router.websocket("/{run_id}/stream")
async def simulation_stream(websocket: WebSocket, run_id: int):
    """
    Bidirectional WebSocket endpoint for real-time Simulation Engine telemetry and snapshots.
    Subscribes directly to Redis pub/sub channel for high-throughput streaming without database overhead.
    """
    await websocket.accept()
    
    pubsub = redis_client.pubsub()
    # Channel name matches SimulationService.get_pubsub_channel
    channel_name = f"simulation_{run_id}"
    await pubsub.subscribe(channel_name)

    async def reader(ws: WebSocket):
        try:
            while True:
                data = await ws.receive_json()
                command = data.get("command")
                if command == "PAUSE":
                    pass
                elif command == "RESUME":
                    pass
        except WebSocketDisconnect:
            pass

    async def writer(ws: WebSocket):
        try:
            async for message in pubsub.listen():
                if message["type"] == "message":
                    payload = message["data"]
                    if isinstance(payload, bytes):
                        payload = payload.decode('utf-8')
                    await ws.send_text(payload)
        except Exception:
            pass
        finally:
            await pubsub.unsubscribe(channel_name)

    read_task = asyncio.create_task(reader(websocket))
    write_task = asyncio.create_task(writer(websocket))
    
    done, pending = await asyncio.wait(
        [read_task, write_task],
        return_when=asyncio.FIRST_COMPLETED
    )
    
    for task in pending:
        task.cancel()
