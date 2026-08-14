import sys, asyncio
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from pathlib import Path


# Ensure the app module can be imported
sys.path.append(str(Path(__file__).parent.parent))

from app.database.session import AsyncSessionLocal
from app.schemas.scenario import ScenarioCreate, ScenarioEventCreate
from app.models.scenario import EventCategory, EventStatus
from app.services.scenario import ScenarioService
import structlog

logger = structlog.get_logger()

async def seed_scenarios():
    async with AsyncSessionLocal() as db:
        service = ScenarioService(db)

        # 1. Global Pandemic
        pandemic_scenario = await service.create_scenario(ScenarioCreate(
            title="Global Pandemic (COVID-19 Analog)",
            description="A highly contagious virus halts global supply chains and forces national lockdowns.",
            is_template=True,
            tags=["Health", "Global", "Supply Chain"]
        ))
        
        evt_lockdown = await service.create_event(pandemic_scenario.id, ScenarioEventCreate(
            title="National Lockdowns",
            category=EventCategory.HEALTH,
            status=EventStatus.ACTIVE,
            parameters={"stringency_index": 80, "duration_months": 6},
            shockIntensity=0.8,
            affectedNodes=["USA", "CHN", "DEU"]
        ))

        evt_supply = await service.create_event(pandemic_scenario.id, ScenarioEventCreate(
            title="Supply Chain Disruption",
            category=EventCategory.SUPPLY_CHAIN,
            status=EventStatus.ACTIVE,
            parent_event_id=evt_lockdown.id,
            parameters={"port_capacity_reduction": 0.4},
            shockIntensity=0.6,
            affectedNodes=["CHN", "USA"]
        ))

        # 2. 2008 Financial Crisis
        financial_scenario = await service.create_scenario(ScenarioCreate(
            title="2008 Financial Crisis",
            description="Subprime mortgage crisis leading to global banking collapse.",
            is_template=True,
            tags=["Financial", "Historical", "Banking"]
        ))

        evt_housing = await service.create_event(financial_scenario.id, ScenarioEventCreate(
            title="Housing Market Crash",
            category=EventCategory.ECONOMIC,
            status=EventStatus.ACTIVE,
            parameters={"asset_price_drop": 0.3},
            shockIntensity=0.9,
            affectedNodes=["USA"]
        ))

        await service.create_event(financial_scenario.id, ScenarioEventCreate(
            title="Bank Failures",
            category=EventCategory.FINANCIAL,
            status=EventStatus.ACTIVE,
            parent_event_id=evt_housing.id,
            parameters={"liquidity_crisis": True, "bankruptcies": 150},
            shockIntensity=0.85,
            affectedNodes=["USA", "GBR", "DEU"]
        ))

        # 3. Russia-Ukraine Conflict
        war_scenario = await service.create_scenario(ScenarioCreate(
            title="Russia-Ukraine Conflict",
            description="Regional war causing energy and food supply shocks.",
            is_template=True,
            tags=["Political", "Energy", "War"]
        ))

        evt_invasion = await service.create_event(war_scenario.id, ScenarioEventCreate(
            title="Military Invasion",
            category=EventCategory.POLITICAL,
            status=EventStatus.ACTIVE,
            parameters={"intensity": "High", "troops": 200000},
            shockIntensity=0.9,
            affectedNodes=["RUS", "UKR"]
        ))

        await service.create_event(war_scenario.id, ScenarioEventCreate(
            title="Energy Sanctions",
            category=EventCategory.ENERGY,
            status=EventStatus.ACTIVE,
            parent_event_id=evt_invasion.id,
            parameters={"gas_supply_cut": 0.8},
            shockIntensity=0.7,
            affectedNodes=["DEU", "FRA", "RUS"]
        ))

        logger.info("Successfully seeded Scenario Templates.")

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.run(seed_scenarios(), loop_factory=asyncio.SelectorEventLoop)
    else:
        asyncio.run(seed_scenarios())



