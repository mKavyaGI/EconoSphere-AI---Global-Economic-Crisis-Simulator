import json
import asyncio
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.models.simulation import SimulationRun, SimulationSnapshot, SimulationStatus, SimulationLog, LogSeverity
from app.engine.compiler import ScenarioCompiler
from app.engine.events import EventProcessor
from app.engine.resolver import DependencyResolver
from app.engine.strategies.manager import StrategyManager
from app.engine.strategies.trade import TradeStrategy
from app.engine.calculators.manager import CalculatorManager
from app.engine.calculators.gdp import GDPCalculator
from app.engine.timeline import TimelineGenerator
from app.engine.recovery import RecoveryEngine
from app.repositories.graph import GraphRepository
from app.schemas.simulation import SimulationConfig


class SimulationEngine:
    """The central orchestrator of the Global Economic Simulation."""

    def __init__(self, session: AsyncSession, run_id: int):
        self.session = session
        self.run_id = run_id
        self.compiler = ScenarioCompiler(session)
        self.event_processor = EventProcessor()
        self.dependency_resolver = DependencyResolver(GraphRepository())
        
        self.strategy_manager = StrategyManager()
        self.strategy_manager.register_strategy(TradeStrategy())
        
        self.calculator_manager = CalculatorManager()
        self.calculator_manager.register_calculator(GDPCalculator())
        
        self.recovery_engine = RecoveryEngine()

    async def _log(self, message: str, component: str = "SimulationEngine", severity: LogSeverity = LogSeverity.INFO):
        log = SimulationLog(
            run_id=self.run_id,
            component=component,
            message=message,
            severity=severity
        )
        self.session.add(log)
        await self.session.commit()

    async def execute(self):
        """Runs the entire simulation pipeline."""
        try:
            # 1. Fetch Run & Config
            stmt = select(SimulationRun).where(SimulationRun.id == self.run_id)
            result = await self.session.execute(stmt)
            run = result.scalar_one_or_none()
            if not run:
                return

            run.status = SimulationStatus.RUNNING
            await self.session.commit()
            
            config = SimulationConfig(**run.config)
            await self._log("Simulation Engine started.", severity=LogSeverity.INFO)

            # 2. Compilation
            baseline_state = await self.compiler.fetch_baseline_state()
            events = await self.compiler.fetch_scenario_events(version_id=run.scenario_id) # Using scenario_id as version for now

            # 3. Event Processing
            initial_shocks = self.event_processor.process(events)
            
            # 4. Dependency Resolution (Mock depth)
            dependency_map = await self.dependency_resolver.resolve_propagation_paths(initial_shocks, max_depth=config.max_depth)

            # 5. Timeline Generation
            timeline = TimelineGenerator(config.time_horizons)
            
            for days in timeline.generate_horizons():
                await self._log(f"Processing Day {days}...", component="TimelineEngine")
                
                # Apply time delay to shocks
                current_shocks = timeline.apply_time_lag(initial_shocks, days)

                # Strategy Execution
                propagated_shocks = self.strategy_manager.apply_all(baseline_state, current_shocks, run.config, dependency_map=dependency_map)
                
                # Calculation
                updated_state = self.calculator_manager.calculate_all(baseline_state, propagated_shocks)
                
                # Recovery
                final_state = self.recovery_engine.apply_recovery(updated_state, days)
                
                # Create Snapshot
                snapshot = SimulationSnapshot(
                    run_id=self.run_id,
                    horizon_days=days,
                    global_state=final_state,
                    active_shocks=propagated_shocks
                )
                self.session.add(snapshot)
                await self.session.commit()
                
                # Here we could publish to Redis for WebSockets
                import app.redis.client
                payload = {
                    "type": "SNAPSHOT",
                    "run_id": self.run_id,
                    "horizon": days,
                    "message": f"Snapshot generated for day {days}",
                    "global_state": final_state,
                    "active_shocks": propagated_shocks
                }
                await app.redis.client.redis_client.publish(f"simulation_{self.run_id}", json.dumps(payload))

            run.status = SimulationStatus.COMPLETED
            run.completed_at = datetime.utcnow()
            await self.session.commit()
            await self._log("Simulation completed successfully.", severity=LogSeverity.INFO)

        except Exception as e:
            await self._log(f"Simulation failed: {str(e)}", severity=LogSeverity.ERROR)
            if 'run' in locals() and run:
                run.status = SimulationStatus.FAILED
                run.completed_at = datetime.utcnow()
                await self.session.commit()
            raise
