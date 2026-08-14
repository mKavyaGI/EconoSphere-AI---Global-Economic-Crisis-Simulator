from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
from app.api.dependencies.database import get_db
from app.schemas.country import CountryResponse, PaginatedCountryResponse, HistoricalIndicatorSchema
from app.services.country import country_service

router = APIRouter()

@router.get("", response_model=PaginatedCountryResponse)
async def get_countries(
    db: AsyncSession = Depends(get_db),
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    region: Optional[str] = None
):
    countries, total = await country_service.get_countries(db, offset=offset, limit=limit, region=region)
    return PaginatedCountryResponse(
        data=countries,
        total=total,
        limit=limit,
        offset=offset
    )

@router.get("/search", response_model=List[CountryResponse])
async def search_countries(
    query: str,
    db: AsyncSession = Depends(get_db)
):
    if len(query) < 2:
        return []
    return await country_service.search_countries(db, query=query)

@router.get("/{iso3}", response_model=CountryResponse)
async def get_country(
    iso3: str,
    db: AsyncSession = Depends(get_db)
):
    country = await country_service.get_country_by_iso3(db, iso3=iso3.upper())
    if not country:
        raise HTTPException(status_code=404, detail="Country not found")
    return country

@router.get("/{iso3}/history", response_model=List[HistoricalIndicatorSchema])
async def get_country_history(
    iso3: str,
    indicator: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    country = await country_service.get_country_by_iso3(db, iso3=iso3.upper())
    if not country:
        raise HTTPException(status_code=404, detail="Country not found")
    return await country_service.get_country_history(db, iso3=iso3.upper(), indicator=indicator)
