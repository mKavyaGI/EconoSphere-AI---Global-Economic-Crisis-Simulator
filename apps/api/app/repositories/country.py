from typing import Optional, List, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, or_
from app.models.country import Country, HistoricalIndicator
from app.repositories.base import BaseRepository
from app.schemas.country import CountryBase

class CountryRepository(BaseRepository[Country, CountryBase, CountryBase]):
    def __init__(self):
        super().__init__(Country)

    async def get_paginated(
        self, db: AsyncSession, *, offset: int = 0, limit: int = 50, region: Optional[str] = None
    ) -> Tuple[List[Country], int]:
        stmt = select(self.model)
        count_stmt = select(func.count()).select_from(self.model)
        
        if region:
            stmt = stmt.where(self.model.region == region)
            count_stmt = count_stmt.where(self.model.region == region)

        stmt = stmt.offset(offset).limit(limit)
        
        total = await db.scalar(count_stmt)
        result = await db.execute(stmt)
        return list(result.scalars().all()), total or 0

    async def search(self, db: AsyncSession, *, query: str, limit: int = 10) -> List[Country]:
        search_term = f"%{query}%"
        stmt = select(self.model).where(
            or_(
                self.model.name.ilike(search_term),
                self.model.iso3.ilike(search_term)
            )
        ).limit(limit)
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def get_history(self, db: AsyncSession, iso3: str, indicator: Optional[str] = None) -> List[HistoricalIndicator]:
        stmt = select(HistoricalIndicator).where(HistoricalIndicator.iso3 == iso3)
        if indicator:
            stmt = stmt.where(HistoricalIndicator.indicator_name == indicator)
        stmt = stmt.order_by(HistoricalIndicator.year.asc())
        result = await db.execute(stmt)
        return list(result.scalars().all())

country_repo = CountryRepository()
