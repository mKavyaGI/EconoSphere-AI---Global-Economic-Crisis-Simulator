from typing import List, Optional, Any, Dict
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import desc

from app.models.simulation import SimulationRun, SimulationSnapshot, SimulationLog, SimulationStatus, LogSeverity


class SimulationRepository:
    """
    Repository layer for managing relational simulation data in PostgreSQL.
    Encapsulates raw database access and session transaction scoping.
    """
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_run(self, scenario_id: int, config: Dict[str, Any]) -> SimulationRun:
        """Creates and initiates a new simulation execution record."""
        run = SimulationRun(
            scenario_id=scenario_id,
            config=config,
            status=SimulationStatus.PENDING,
            started_at=datetime.now(timezone.utc)
        )
        self.session.add(run)
        await self.session.commit()
        await self.session.refresh(run)
        return run

    async def get_run(self, run_id: int) -> Optional[SimulationRun]:
        """Retrieves a simulation run by ID with its loaded relationships."""
        stmt = select(SimulationRun).where(SimulationRun.id == run_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_runs(self, scenario_id: Optional[int] = None, limit: int = 50, offset: int = 0) -> List[SimulationRun]:
        """Lists simulation runs with optional filtering by scenario_id and pagination."""
        stmt = select(SimulationRun).order_by(desc(SimulationRun.started_at)).limit(limit).offset(offset)
        if scenario_id is not None:
            stmt = stmt.where(SimulationRun.scenario_id == scenario_id)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def update_status(self, run_id: int, status: SimulationStatus, completed_at: Optional[datetime] = None) -> Optional[SimulationRun]:
        """Updates the status and optional completion timestamp of a simulation run."""
        run = await self.get_run(run_id)
        if not run:
            return None
        run.status = status
        if completed_at:
            run.completed_at = completed_at
        elif status in [SimulationStatus.COMPLETED, SimulationStatus.FAILED, SimulationStatus.PAUSED]:
            if status in [SimulationStatus.COMPLETED, SimulationStatus.FAILED]:
                run.completed_at = datetime.now(timezone.utc)
        self.session.add(run)
        await self.session.commit()
        await self.session.refresh(run)
        return run

    async def delete_run(self, run_id: int) -> bool:
        """Deletes a simulation run and all associated snapshots and logs (via cascade)."""
        run = await self.get_run(run_id)
        if not run:
            return False
        await self.session.delete(run)
        await self.session.commit()
        return True

    # Snapshots
    async def add_snapshot(self, run_id: int, horizon_days: int, global_state: Dict[str, Any], active_shocks: Dict[str, Any]) -> SimulationSnapshot:
        """Persists a new simulation snapshot at a given timeline horizon."""
        snapshot = SimulationSnapshot(
            run_id=run_id,
            horizon_days=horizon_days,
            timestamp=datetime.now(timezone.utc),
            global_state=global_state,
            active_shocks=active_shocks
        )
        self.session.add(snapshot)
        await self.session.commit()
        await self.session.refresh(snapshot)
        return snapshot

    async def get_snapshots(self, run_id: int) -> List[SimulationSnapshot]:
        """Retrieves timeline snapshots for a simulation run ordered by horizon days."""
        stmt = select(SimulationSnapshot).where(SimulationSnapshot.run_id == run_id).order_by(SimulationSnapshot.horizon_days)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    # Logs
    async def add_log(
        self,
        run_id: int,
        component: str,
        message: str,
        severity: LogSeverity = LogSeverity.INFO,
        affected_country: Optional[str] = None,
        affected_indicator: Optional[str] = None
    ) -> SimulationLog:
        """Records an audit or execution log entry for a simulation run."""
        log_entry = SimulationLog(
            run_id=run_id,
            timestamp=datetime.now(timezone.utc),
            component=component,
            message=message,
            severity=severity,
            affected_country=affected_country,
            affected_indicator=affected_indicator
        )
        self.session.add(log_entry)
        await self.session.commit()
        await self.session.refresh(log_entry)
        return log_entry

    async def get_logs(self, run_id: int, severity: Optional[LogSeverity] = None) -> List[SimulationLog]:
        """Retrieves logs for a specific simulation run with optional severity filtering."""
        stmt = select(SimulationLog).where(SimulationLog.run_id == run_id).order_by(SimulationLog.timestamp)
        if severity is not None:
            stmt = stmt.where(SimulationLog.severity == severity)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
