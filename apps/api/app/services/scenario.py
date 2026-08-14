from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from app.repositories.scenario import ScenarioRepository
from app.schemas.scenario import (
    ScenarioCreate, ScenarioUpdate, Scenario, 
    ScenarioVersionCreate, ScenarioEventCreate, ScenarioEventUpdate, ScenarioEvent
)
from app.models.scenario import EventStatus, ScenarioStatus

class ScenarioService:
    def __init__(self, session: AsyncSession):
        self.repo = ScenarioRepository(session)
        self.session = session

    async def get_scenarios(
        self, 
        is_template: Optional[bool] = None,
        status: Optional[str] = None,
        title: Optional[str] = None,
        skip: int = 0,
        limit: int = 50
    ) -> List[Scenario]:
        return await self.repo.get_scenarios(
            is_template=is_template,
            status=status,
            title=title,
            skip=skip,
            limit=limit
        )
        
    async def get_scenario(self, scenario_id: int) -> Optional[Scenario]:
        return await self.repo.get_scenario(scenario_id)

    async def create_scenario(self, scenario_in: ScenarioCreate) -> Scenario:
        # Create scenario
        scenario = await self.repo.create_scenario(scenario_in)
        # Create initial version 1
        await self.repo.create_version(scenario.id, ScenarioVersionCreate(version_number=1))
        # Refresh to include versions
        return await self.get_scenario(scenario.id)

    async def update_scenario(self, scenario_id: int, scenario_in: ScenarioUpdate) -> Optional[Scenario]:
        return await self.repo.update_scenario(scenario_id, scenario_in)

    async def duplicate_scenario(self, scenario_id: int, new_title: str) -> Optional[Scenario]:
        original = await self.repo.get_scenario(scenario_id)
        if not original:
            return None
            
        # Create duplicate scenario shell
        new_scenario_in = ScenarioCreate(
            title=new_title,
            description=original.description,
            is_template=False,
            tags=original.tags
        )
        new_scenario = await self.repo.create_scenario(new_scenario_in)
        
        latest_version = await self.repo.get_latest_version(scenario_id)
        new_version = await self.repo.create_version(new_scenario.id, ScenarioVersionCreate(version_number=1))
        
        if latest_version and latest_version.events:
            # Two-pass copy to preserve parent-child tree
            # Pass 1: Copy root events (no parent), build old_id -> new_id map
            old_to_new_id: dict[int, int] = {}
            for event in latest_version.events:
                if event.parent_event_id is None:
                    event_in = ScenarioEventCreate(
                        title=event.title,
                        category=event.category,
                        status=event.status,
                        parameters=event.parameters,
                        shockIntensity=event.shockIntensity,
                        propagationDelay=event.propagationDelay,
                        recoveryRate=event.recoveryRate,
                        affectedNodes=event.affectedNodes,
                        confidence=event.confidence,
                        simulationWeight=event.simulationWeight,
                        parent_event_id=None
                    )
                    new_event = await self.repo.create_event(new_version.id, event_in)
                    old_to_new_id[event.id] = new_event.id

            # Pass 2: Copy child events, remapping parent IDs
            for event in latest_version.events:
                if event.parent_event_id is not None:
                    new_parent_id = old_to_new_id.get(event.parent_event_id)
                    event_in = ScenarioEventCreate(
                        title=event.title,
                        category=event.category,
                        status=event.status,
                        parameters=event.parameters,
                        shockIntensity=event.shockIntensity,
                        propagationDelay=event.propagationDelay,
                        recoveryRate=event.recoveryRate,
                        affectedNodes=event.affectedNodes,
                        confidence=event.confidence,
                        simulationWeight=event.simulationWeight,
                        parent_event_id=new_parent_id
                    )
                    new_event = await self.repo.create_event(new_version.id, event_in)
                    old_to_new_id[event.id] = new_event.id
                    
        return await self.get_scenario(new_scenario.id)

    async def delete_scenario(self, scenario_id: int) -> bool:
        return await self.repo.delete_scenario(scenario_id)

    async def create_event(self, scenario_id: int, event_in: ScenarioEventCreate) -> Optional[ScenarioEvent]:
        # Always add to latest version
        latest_version = await self.repo.get_latest_version(scenario_id)
        if not latest_version:
            return None
        return await self.repo.create_event(latest_version.id, event_in)

    async def update_event(self, scenario_id: int, event_id: int, event_in: ScenarioEventUpdate) -> Optional[ScenarioEvent]:
        # In a strict event-sourcing model, this would create a new version.
        # For this MVP, we mutate the latest version in place.
        return await self.repo.update_event(event_id, event_in)

    async def delete_event(self, scenario_id: int, event_id: int) -> bool:
        return await self.repo.delete_event(event_id)

    # Added features for Economic Scenario Studio
    async def validate_scenario(self, scenario_id: int) -> dict:
        scenario = await self.repo.get_scenario(scenario_id)
        if not scenario:
            return {"is_valid": False, "errors": ["Scenario not found."], "warnings": []}
            
        latest_version = await self.repo.get_latest_version(scenario_id)
        if not latest_version or not latest_version.events:
            return {
                "is_valid": False, 
                "errors": ["Scenario has no active events or versions defined."], 
                "warnings": ["An empty scenario cannot be evaluated in the simulation engine."]
            }

        errors = []
        warnings = []
        event_ids = {e.id for e in latest_version.events}

        for event in latest_version.events:
            # Check title
            if not event.title or len(event.title.strip()) == 0:
                errors.append(f"Event ID {event.id} is missing a valid title.")

            # Check shock intensity bounds
            if event.shockIntensity is not None:
                if not (0.0 <= event.shockIntensity <= 1.0):
                    errors.append(f"Event '{event.title}' has shockIntensity ({event.shockIntensity}) outside valid range [0.0, 1.0].")
            else:
                warnings.append(f"Event '{event.title}' has default or unassigned shock intensity.")

            # Check parent references & circularity
            if event.parent_event_id is not None:
                if event.parent_event_id not in event_ids:
                    errors.append(f"Event '{event.title}' references nonexistent parent event ID {event.parent_event_id}.")
                elif event.parent_event_id == event.id:
                    errors.append(f"Event '{event.title}' has a circular reference to itself as parent.")

            # Check affected nodes
            if not event.affectedNodes or len(event.affectedNodes) == 0:
                warnings.append(f"Event '{event.title}' does not specify any affected nodes (countries/sectors).")

        is_valid = len(errors) == 0
        if is_valid:
            warnings.append("Simulation ready: valid parameter boundaries and causal relationships verified.")

        return {"is_valid": is_valid, "errors": errors, "warnings": warnings}

    async def publish_scenario(self, scenario_id: int) -> Optional[Scenario]:
        """Transitions scenario to PUBLISHED status."""
        db_obj = await self.repo.get_scenario(scenario_id)
        if not db_obj:
            return None
        db_obj.status = ScenarioStatus.PUBLISHED
        await self.session.commit()
        await self.session.refresh(db_obj)
        return await self.get_scenario(scenario_id)

    async def archive_scenario(self, scenario_id: int) -> Optional[Scenario]:
        """Transitions scenario to ARCHIVED status."""
        db_obj = await self.repo.get_scenario(scenario_id)
        if not db_obj:
            return None
        db_obj.status = ScenarioStatus.ARCHIVED
        await self.session.commit()
        await self.session.refresh(db_obj)
        return await self.get_scenario(scenario_id)

    async def restore_scenario(self, scenario_id: int) -> Optional[Scenario]:
        """Restores an archived scenario back to DRAFT status."""
        db_obj = await self.repo.get_scenario(scenario_id)
        if not db_obj:
            return None
        db_obj.status = ScenarioStatus.DRAFT
        await self.session.commit()
        await self.session.refresh(db_obj)
        return await self.get_scenario(scenario_id)
