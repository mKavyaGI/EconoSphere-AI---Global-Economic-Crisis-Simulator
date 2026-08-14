from typing import Optional, List, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.country import Country, HistoricalIndicator
from app.repositories.country import country_repo
from app.services.base import BaseService

class CountryService(BaseService[type(country_repo)]):
    def __init__(self):
        super().__init__(country_repo)

    async def get_countries(
        self, db: AsyncSession, offset: int = 0, limit: int = 50, region: Optional[str] = None
    ) -> Tuple[List[Country], int]:
        return await self.repository.get_paginated(db, offset=offset, limit=limit, region=region)

    async def search_countries(self, db: AsyncSession, query: str) -> List[Country]:
        return await self.repository.search(db, query=query)

    async def get_country_by_iso3(self, db: AsyncSession, iso3: str) -> Optional[Country]:
        return await self.repository.get(db, iso3)

    async def get_country_history(self, db: AsyncSession, iso3: str, indicator: Optional[str] = None) -> List[HistoricalIndicator]:
        return await self.repository.get_history(db, iso3, indicator)

country_service = CountryService()
