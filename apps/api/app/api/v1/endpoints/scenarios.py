from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.api.dependencies.database import get_db
from app.schemas.scenario import Scenario, ScenarioCreate, ScenarioUpdate, ScenarioEvent, ScenarioEventCreate, ScenarioEventUpdate
from app.services.scenario import ScenarioService
from app.models.scenario import EventCategory

router = APIRouter()

@router.get("/", response_model=List[Scenario])
async def get_scenarios(
    is_template: Optional[bool] = None, 
    status: Optional[str] = None,
    title: Optional[str] = None,
    skip: int = 0,
    limit: int = 50,
    db: AsyncSession = Depends(get_db)
):
    service = ScenarioService(db)
    return await service.get_scenarios(
        is_template=is_template,
        status=status,
        title=title,
        skip=skip,
        limit=limit
    )

@router.get("/templates", response_model=List[Scenario])
async def get_templates(
    skip: int = 0,
    limit: int = 50,
    db: AsyncSession = Depends(get_db)
):
    service = ScenarioService(db)
    return await service.get_scenarios(is_template=True, skip=skip, limit=limit)

@router.get("/events/types")
async def get_event_types():
    return [e.value for e in EventCategory]

@router.get("/{scenario_id}", response_model=Scenario)
async def get_scenario(scenario_id: int, db: AsyncSession = Depends(get_db)):
    service = ScenarioService(db)
    scenario = await service.get_scenario(scenario_id)
    if not scenario:
        raise HTTPException(status_code=404, detail="Scenario not found")
    return scenario

@router.post("/", response_model=Scenario)
async def create_scenario(scenario_in: ScenarioCreate, db: AsyncSession = Depends(get_db)):
    service = ScenarioService(db)
    return await service.create_scenario(scenario_in)

@router.put("/{scenario_id}", response_model=Scenario)
async def update_scenario(scenario_id: int, scenario_in: ScenarioUpdate, db: AsyncSession = Depends(get_db)):
    service = ScenarioService(db)
    scenario = await service.update_scenario(scenario_id, scenario_in)
    if not scenario:
        raise HTTPException(status_code=404, detail="Scenario not found")
    return scenario

@router.delete("/{scenario_id}")
async def delete_scenario(scenario_id: int, db: AsyncSession = Depends(get_db)):
    service = ScenarioService(db)
    success = await service.delete_scenario(scenario_id)
    if not success:
        raise HTTPException(status_code=404, detail="Scenario not found")
    return {"message": "Scenario deleted successfully"}

@router.post("/{scenario_id}/duplicate", response_model=Scenario)
async def duplicate_scenario(scenario_id: int, new_title: str, db: AsyncSession = Depends(get_db)):
    service = ScenarioService(db)
    scenario = await service.duplicate_scenario(scenario_id, new_title)
    if not scenario:
        raise HTTPException(status_code=404, detail="Scenario not found")
    return scenario

@router.post("/{scenario_id}/validate")
async def validate_scenario(scenario_id: int, db: AsyncSession = Depends(get_db)):
    service = ScenarioService(db)
    return await service.validate_scenario(scenario_id)

@router.post("/{scenario_id}/publish", response_model=Scenario)
async def publish_scenario(scenario_id: int, db: AsyncSession = Depends(get_db)):
    service = ScenarioService(db)
    scenario = await service.publish_scenario(scenario_id)
    if not scenario:
        raise HTTPException(status_code=404, detail="Scenario not found")
    return scenario

@router.post("/{scenario_id}/archive", response_model=Scenario)
async def archive_scenario(scenario_id: int, db: AsyncSession = Depends(get_db)):
    service = ScenarioService(db)
    scenario = await service.archive_scenario(scenario_id)
    if not scenario:
        raise HTTPException(status_code=404, detail="Scenario not found")
    return scenario

@router.post("/{scenario_id}/restore", response_model=Scenario)
async def restore_scenario(scenario_id: int, db: AsyncSession = Depends(get_db)):
    service = ScenarioService(db)
    scenario = await service.restore_scenario(scenario_id)
    if not scenario:
        raise HTTPException(status_code=404, detail="Scenario not found")
    return scenario

@router.post("/{scenario_id}/events", response_model=ScenarioEvent)
async def create_event(scenario_id: int, event_in: ScenarioEventCreate, db: AsyncSession = Depends(get_db)):
    service = ScenarioService(db)
    event = await service.create_event(scenario_id, event_in)
    if not event:
        raise HTTPException(status_code=404, detail="Scenario not found or no versions exist")
    return event

@router.put("/{scenario_id}/events/{event_id}", response_model=ScenarioEvent)
async def update_event(scenario_id: int, event_id: int, event_in: ScenarioEventUpdate, db: AsyncSession = Depends(get_db)):
    service = ScenarioService(db)
    event = await service.update_event(scenario_id, event_id, event_in)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    return event

@router.delete("/{scenario_id}/events/{event_id}")
async def delete_event(scenario_id: int, event_id: int, db: AsyncSession = Depends(get_db)):
    service = ScenarioService(db)
    success = await service.delete_event(scenario_id, event_id)
    if not success:
        raise HTTPException(status_code=404, detail="Event not found")
    return {"message": "Event deleted successfully"}
