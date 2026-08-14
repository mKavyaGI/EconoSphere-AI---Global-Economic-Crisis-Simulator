from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from rq import Queue

from app.models.simulation import SimulationRun, SimulationSnapshot, SimulationLog, SimulationStatus, LogSeverity
from app.repositories.simulation import SimulationRepository
from app.repositories.scenario import ScenarioRepository
from app.repositories.graph import GraphRepository
from app.redis.client import sync_redis_client
from workers.simulation_worker import run_simulation_job

logger = structlog.get_logger()


class ISimulationService(ABC):
    """
    Abstract interface defining the business logic contract for simulation orchestration.
    Ensures seamless dependency injection and future extensibility for AI agent integrations in Phase 11+.
    """
    @abstractmethod
    async def start_simulation(self, scenario_id: int, config_data: Optional[Dict[str, Any]] = None) -> SimulationRun:
        pass

    @abstractmethod
    async def get_simulation_status(self, run_id: int) -> Optional[SimulationRun]:
        pass

    @abstractmethod
    async def stop_simulation(self, run_id: int) -> Optional[SimulationRun]:
        pass

    @abstractmethod
    async def list_simulation_runs(self, scenario_id: Optional[int] = None, limit: int = 50, offset: int = 0) -> List[SimulationRun]:
        pass

    @abstractmethod
    async def get_run_snapshots(self, run_id: int) -> List[SimulationSnapshot]:
        pass

    @abstractmethod
    async def get_run_logs(self, run_id: int, severity: Optional[LogSeverity] = None) -> List[SimulationLog]:
        pass

    @abstractmethod
    def get_pubsub_channel(self, run_id: int) -> str:
        pass


class SimulationService(ISimulationService):
    """
    Concrete orchestration service for EconoSphere AI Simulation Engine.
    Coordinates PostgreSQL repositories, Neo4j topological graph queries, Redis RQ workers, and transaction boundaries.
    """
    def __init__(
        self,
        session: AsyncSession,
        sim_repo: Optional[SimulationRepository] = None,
        scenario_repo: Optional[ScenarioRepository] = None,
        graph_repo: Optional[GraphRepository] = None,
        queue: Optional[Queue] = None
    ):
        self.session = session
        self.sim_repo = sim_repo or SimulationRepository(session)
        self.scenario_repo = scenario_repo or ScenarioRepository(session)
        self.graph_repo = graph_repo or GraphRepository()
        self.queue = queue or Queue('simulation_queue', connection=sync_redis_client)

    async def start_simulation(self, scenario_id: int, config_data: Optional[Dict[str, Any]] = None) -> SimulationRun:
        """
        Validates scenario existence, performs topological readiness checks against Neo4j,
        creates a transaction-scoped database execution record, and dispatches an asynchronous worker job.
        """
        config = config_data or {}
        logger.info("initiating_simulation_run", scenario_id=scenario_id, config=config)
        
        try:
            # 1. Validate scenario exists
            scenario = await self.scenario_repo.get_scenario(scenario_id)
            if not scenario:
                logger.error("simulation_start_failed_missing_scenario", scenario_id=scenario_id)
                raise ValueError(f"Scenario with id {scenario_id} does not exist.")

            # 2. Check Neo4j Graph Network Readiness (Log warning if graph unavailable, fallback will handle)
            try:
                graph_test = await self.graph_repo.run_query("MATCH (n:Country) RETURN count(n) AS count")
                logger.info("neo4j_graph_verified_ready_for_simulation", nodes=graph_test)
            except Exception as e:
                logger.warning("neo4j_graph_unavailable_for_simulation_precheck", error=str(e))

            # 3. Persist simulation run in PostgreSQL within transaction
            run = await self.sim_repo.create_run(scenario_id=scenario_id, config=config)

            # 4. Record audit log
            await self.sim_repo.add_log(
                run_id=run.id,
                component="Orchestrator",
                message=f"Simulation run initialized for scenario '{scenario.title}'. Enqueueing job to Redis.",
                severity=LogSeverity.INFO
            )

            # 5. Dispatch async RQ job to worker
            try:
                self.queue.enqueue(run_simulation_job, run.id)
                logger.info("simulation_job_enqueued", run_id=run.id, queue_name=self.queue.name)
            except Exception as e:
                logger.error("redis_enqueue_failed", run_id=run.id, error=str(e))
                await self.sim_repo.update_status(run_id=run.id, status=SimulationStatus.FAILED)
                await self.sim_repo.add_log(
                    run_id=run.id,
                    component="Orchestrator",
                    message=f"Failed to enqueue RQ job: {str(e)}",
                    severity=LogSeverity.ERROR
                )
                raise RuntimeError(f"Failed to enqueue simulation job: {str(e)}")

            return run

        except ValueError:
            raise
        except Exception as e:
            logger.error("unexpected_error_starting_simulation", error=str(e))
            await self.session.rollback()
            raise

    async def get_simulation_status(self, run_id: int) -> Optional[SimulationRun]:
        """Queries current status, configuration, and timestamps for a run."""
        return await self.sim_repo.get_run(run_id)

    async def stop_simulation(self, run_id: int) -> Optional[SimulationRun]:
        """Terminates or pauses an active simulation run and logs the administrative intervention."""
        run = await self.sim_repo.get_run(run_id)
        if not run:
            return None

        logger.info("stopping_simulation_run", run_id=run_id, current_status=run.status)
        updated_run = await self.sim_repo.update_status(run_id=run_id, status=SimulationStatus.PAUSED)
        
        await self.sim_repo.add_log(
            run_id=run_id,
            component="Orchestrator",
            message="Simulation run was stopped/paused via client request.",
            severity=LogSeverity.WARNING
        )
        return updated_run

    async def list_simulation_runs(self, scenario_id: Optional[int] = None, limit: int = 50, offset: int = 0) -> List[SimulationRun]:
        """Returns paginated list of simulation runs."""
        return await self.sim_repo.list_runs(scenario_id=scenario_id, limit=limit, offset=offset)

    async def get_run_snapshots(self, run_id: int) -> List[SimulationSnapshot]:
        """Returns timeline propagation snapshots for visualization and analysis."""
        return await self.sim_repo.get_snapshots(run_id=run_id)

    async def get_run_logs(self, run_id: int, severity: Optional[LogSeverity] = None) -> List[SimulationLog]:
        """Returns audit execution logs for a specific run."""
        return await self.sim_repo.get_logs(run_id=run_id, severity=severity)

    def get_pubsub_channel(self, run_id: int) -> str:
        """Returns the targeted Redis pub/sub channel name for real-time WebSocket streaming."""
        return f"simulation_{run_id}"
