from pydantic import BaseModel, ConfigDict
from typing import Optional, List

class CountryMetadataSchema(BaseModel):
    capital: Optional[str] = None
    population: Optional[int] = None
    currency: Optional[str] = None
    flag_url: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None

    model_config = ConfigDict(from_attributes=True)

class CountryBase(BaseModel):
    iso3: str
    name: str
    region: Optional[str] = None
    income_group: Optional[str] = None

class CountryResponse(CountryBase):
    metadata_info: Optional[CountryMetadataSchema] = None

    model_config = ConfigDict(from_attributes=True)

class HistoricalIndicatorSchema(BaseModel):
    indicator_name: str
    year: int
    value: float
    source: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)
    
class PaginatedCountryResponse(BaseModel):
    data: List[CountryResponse]
    total: int
    limit: int
    offset: int
