from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models.scenario import Scenario, ScenarioVersion, ScenarioEvent, EventStatus
from app.schemas.scenario import ScenarioCreate, ScenarioUpdate, ScenarioVersionCreate, ScenarioEventCreate, ScenarioEventUpdate

class ScenarioRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    # Scenarios
    async def get_scenario(self, scenario_id: int) -> Optional[Scenario]:
        result = await self.session.execute(select(Scenario).where(Scenario.id == scenario_id))
        return result.scalar_one_or_none()

    async def get_scenarios(
        self, 
        is_template: Optional[bool] = None,
        status: Optional[str] = None,
        title: Optional[str] = None,
        skip: int = 0,
        limit: int = 50
    ) -> List[Scenario]:
        query = select(Scenario)
        if is_template is not None:
            query = query.where(Scenario.is_template == is_template)
        if status is not None:
            query = query.where(Scenario.status == status)
        if title:
            query = query.where(Scenario.title.ilike(f"%{title}%"))
        query = query.offset(skip).limit(limit)
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def create_scenario(self, scenario_in: ScenarioCreate) -> Scenario:
        db_obj = Scenario(**scenario_in.model_dump())
        self.session.add(db_obj)
        await self.session.commit()
        await self.session.refresh(db_obj)
        return db_obj

    async def update_scenario(self, scenario_id: int, scenario_in: ScenarioUpdate) -> Optional[Scenario]:
        db_obj = await self.get_scenario(scenario_id)
        if db_obj:
            update_data = scenario_in.model_dump(exclude_unset=True)
            for field, value in update_data.items():
                setattr(db_obj, field, value)
            await self.session.commit()
            await self.session.refresh(db_obj)
        return db_obj

    async def delete_scenario(self, scenario_id: int) -> bool:
        db_obj = await self.get_scenario(scenario_id)
        if db_obj:
            await self.session.delete(db_obj)
            await self.session.commit()
            return True
        return False

    # Versions
    async def create_version(self, scenario_id: int, version_in: ScenarioVersionCreate) -> ScenarioVersion:
        db_obj = ScenarioVersion(scenario_id=scenario_id, **version_in.model_dump())
        self.session.add(db_obj)
        await self.session.commit()
        await self.session.refresh(db_obj)
        return db_obj
        
    async def get_latest_version(self, scenario_id: int) -> Optional[ScenarioVersion]:
        result = await self.session.execute(
            select(ScenarioVersion)
            .where(ScenarioVersion.scenario_id == scenario_id)
            .order_by(ScenarioVersion.version_number.desc())
        )
        return result.scalars().first()

    # Events
    async def get_event(self, event_id: int) -> Optional[ScenarioEvent]:
        result = await self.session.execute(select(ScenarioEvent).where(ScenarioEvent.id == event_id))
        return result.scalar_one_or_none()

    async def create_event(self, version_id: int, event_in: ScenarioEventCreate) -> ScenarioEvent:
        db_obj = ScenarioEvent(version_id=version_id, **event_in.model_dump())
        self.session.add(db_obj)
        await self.session.commit()
        await self.session.refresh(db_obj)
        return db_obj

    async def update_event(self, event_id: int, event_in: ScenarioEventUpdate) -> Optional[ScenarioEvent]:
        db_obj = await self.get_event(event_id)
        if db_obj:
            update_data = event_in.model_dump(exclude_unset=True)
            for field, value in update_data.items():
                setattr(db_obj, field, value)
            await self.session.commit()
            await self.session.refresh(db_obj)
        return db_obj

    async def delete_event(self, event_id: int) -> bool:
        db_obj = await self.get_event(event_id)
        if db_obj:
            await self.session.delete(db_obj)
            await self.session.commit()
            return True
        return False
