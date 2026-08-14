import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.session import AsyncSessionLocal
from app.models.country import Country, CountryMetadata, HistoricalIndicator

SEED_COUNTRIES = [
    {"iso3": "USA", "name": "United States", "region": "North America", "income_group": "High income", "cap": "Washington, D.C.", "pop": 331000000, "cur": "USD"},
    {"iso3": "CHN", "name": "China", "region": "East Asia & Pacific", "income_group": "Upper middle income", "cap": "Beijing", "pop": 1402000000, "cur": "CNY"},
    {"iso3": "IND", "name": "India", "region": "South Asia", "income_group": "Lower middle income", "cap": "New Delhi", "pop": 1380000000, "cur": "INR"},
    {"iso3": "DEU", "name": "Germany", "region": "Europe & Central Asia", "income_group": "High income", "cap": "Berlin", "pop": 83240000, "cur": "EUR"},
    {"iso3": "JPN", "name": "Japan", "region": "East Asia & Pacific", "income_group": "High income", "cap": "Tokyo", "pop": 125800000, "cur": "JPY"},
    {"iso3": "GBR", "name": "United Kingdom", "region": "Europe & Central Asia", "income_group": "High income", "cap": "London", "pop": 67220000, "cur": "GBP"},
    {"iso3": "FRA", "name": "France", "region": "Europe & Central Asia", "income_group": "High income", "cap": "Paris", "pop": 67390000, "cur": "EUR"},
    {"iso3": "BRA", "name": "Brazil", "region": "Latin America & Caribbean", "income_group": "Upper middle income", "cap": "Brasilia", "pop": 212600000, "cur": "BRL"},
    {"iso3": "AUS", "name": "Australia", "region": "East Asia & Pacific", "income_group": "High income", "cap": "Canberra", "pop": 25690000, "cur": "AUD"},
    {"iso3": "CAN", "name": "Canada", "region": "North America", "income_group": "High income", "cap": "Ottawa", "pop": 38010000, "cur": "CAD"},
]

async def seed_data():
    async with AsyncSessionLocal() as db:
        for data in SEED_COUNTRIES:
            iso3 = data["iso3"]
            
            # Check if exists
            existing = await db.get(Country, iso3)
            if not existing:
                country = Country(iso3=iso3, name=data["name"], region=data["region"], income_group=data["income_group"])
                db.add(country)
                
                meta = CountryMetadata(
                    iso3=iso3,
                    capital=data["cap"],
                    population=data["pop"],
                    currency=data["cur"]
                )
                db.add(meta)
                
                # Mock 10 years of GDP/Inflation
                for year in range(2014, 2024):
                    db.add(HistoricalIndicator(iso3=iso3, indicator_name="GDP", year=year, value=1.0 + (year - 2014)*0.05, source="World Bank"))
                    db.add(HistoricalIndicator(iso3=iso3, indicator_name="Inflation", year=year, value=2.0 + (year % 3)*0.5, source="World Bank"))
        
        await db.commit()
        print("Successfully seeded 10 countries and historical data.")

if __name__ == "__main__":
    asyncio.run(seed_data())
