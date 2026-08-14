from typing import List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload

from app.models.scenario import ScenarioVersion
from app.models.country import Country
from app.engine.interfaces import GlobalState


class ScenarioCompiler:
    """Fetches baseline data and scenario context before execution."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def fetch_baseline_state(self) -> GlobalState:
        """
        Fetches all countries and their latest indicators to form the baseline state.
        Returns a GlobalState dictionary keyed by ISO code.
        """
        stmt = select(Country).options(selectinload(Country.historical_indicators))
        result = await self.session.execute(stmt)
        countries = result.scalars().all()
        
        state: GlobalState = {}
        for country in countries:
            # Find the latest indicator year for baseline
            if country.historical_indicators:
                latest_indicator = sorted(country.historical_indicators, key=lambda x: x.year, reverse=True)[0]
                state[country.iso3] = {
                    "gdp": latest_indicator.gdp,
                    "inflation": latest_indicator.inflation,
                    "unemployment": latest_indicator.unemployment,
                    "interest_rate": latest_indicator.interest_rate,
                    "trade_balance": latest_indicator.trade_balance,
                }
            else:
                # Default empty baseline if no data
                state[country.iso3] = {
                    "gdp": 0.0,
                    "inflation": 0.0,
                    "unemployment": 0.0,
                    "interest_rate": 0.0,
                    "trade_balance": 0.0,
                }
        return state

    async def fetch_scenario_events(self, version_id: int):
        """Fetches the events attached to a specific scenario version."""
        stmt = select(ScenarioVersion).options(
            selectinload(ScenarioVersion.events)
        ).where(ScenarioVersion.id == version_id)
        result = await self.session.execute(stmt)
        version = result.scalar_one_or_none()
        if not version:
            raise ValueError(f"Scenario version {version_id} not found")
        return version.events
