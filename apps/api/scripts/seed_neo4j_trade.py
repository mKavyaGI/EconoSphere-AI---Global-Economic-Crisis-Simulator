import asyncio
import sys
from pathlib import Path
import random

# Ensure the app module can be imported
sys.path.append(str(Path(__file__).parent.parent))

from app.neo4j.session import neo4j_conn
from app.core.config import settings
import structlog

logger = structlog.get_logger()

COUNTRIES = [
    {"iso3": "USA", "name": "United States", "region": "North America", "income_group": "High income"},
    {"iso3": "CHN", "name": "China", "region": "East Asia & Pacific", "income_group": "Upper middle income"},
    {"iso3": "IND", "name": "India", "region": "South Asia", "income_group": "Lower middle income"},
    {"iso3": "DEU", "name": "Germany", "region": "Europe & Central Asia", "income_group": "High income"},
    {"iso3": "JPN", "name": "Japan", "region": "East Asia & Pacific", "income_group": "High income"},
    {"iso3": "GBR", "name": "United Kingdom", "region": "Europe & Central Asia", "income_group": "High income"},
    {"iso3": "FRA", "name": "France", "region": "Europe & Central Asia", "income_group": "High income"},
    {"iso3": "BRA", "name": "Brazil", "region": "Latin America & Caribbean", "income_group": "Upper middle income"},
    {"iso3": "AUS", "name": "Australia", "region": "East Asia & Pacific", "income_group": "High income"},
    {"iso3": "CAN", "name": "Canada", "region": "North America", "income_group": "High income"}
]

COMMODITIES = [
    {"id": "oil", "name": "Crude Oil", "category": "Energy"},
    {"id": "electronics", "name": "Consumer Electronics", "category": "Technology"},
    {"id": "autos", "name": "Automobiles", "category": "Manufacturing"},
    {"id": "agri", "name": "Agriculture", "category": "Food"},
    {"id": "pharma", "name": "Pharmaceuticals", "category": "Healthcare"}
]

BLOCS = [
    {"id": "g7", "name": "G7", "type": "Economic Forum"},
    {"id": "brics", "name": "BRICS", "type": "Economic Alliance"},
    {"id": "eu", "name": "European Union", "type": "Political & Economic Union"}
]

MEMBERSHIPS = {
    "g7": ["USA", "DEU", "JPN", "GBR", "FRA", "CAN"],
    "brics": ["CHN", "IND", "BRA"],
    "eu": ["DEU", "FRA"]
}

async def clear_database(tx):
    await tx.run("MATCH (n) DETACH DELETE n")

async def create_nodes(tx):
    # Countries
    for c in COUNTRIES:
        await tx.run(
            """
            CREATE (:Country {
                iso3: $iso3, name: $name, region: $region, income_group: $income_group,
                health: $health, stability: $stability, risk: $risk, shockValue: 0.0
            })
            """,
            iso3=c["iso3"], name=c["name"], region=c["region"], income_group=c["income_group"],
            health=random.uniform(0.7, 1.0), stability=random.uniform(0.6, 0.95), risk=random.uniform(0.1, 0.4)
        )
    
    # Commodities
    for cmd in COMMODITIES:
        await tx.run(
            "CREATE (:Commodity {id: $id, name: $name, category: $category})",
            id=cmd["id"], name=cmd["name"], category=cmd["category"]
        )
        
    # Blocs
    for b in BLOCS:
        await tx.run(
            "CREATE (:EconomicBloc {id: $id, name: $name, type: $type})",
            id=b["id"], name=b["name"], type=b["type"]
        )

async def create_relationships(tx):
    # Bloc Memberships
    for bloc_id, members in MEMBERSHIPS.items():
        for iso3 in members:
            await tx.run(
                """
                MATCH (c:Country {iso3: $iso3}), (b:EconomicBloc {id: $bloc_id})
                CREATE (c)-[:MEMBER_OF {joined_year: 2000}]->(b)
                """,
                iso3=iso3, bloc_id=bloc_id
            )

    # Mock Trade Routes (EXPORTS_TO)
    years = [2022, 2023]
    for source in COUNTRIES:
        # Each country exports to 3-5 random other countries
        targets = random.sample([c for c in COUNTRIES if c["iso3"] != source["iso3"]], k=random.randint(3, 5))
        for target in targets:
            for commodity in random.sample(COMMODITIES, k=random.randint(1, 3)):
                for year in years:
                    volume = random.uniform(10_000, 500_000)
                    dependency = random.uniform(0.1, 0.8)
                    growth = random.uniform(-0.05, 0.15)
                    await tx.run(
                        """
                        MATCH (s:Country {iso3: $source}), (t:Country {iso3: $target})
                        CREATE (s)-[:EXPORTS_TO {
                            commodity_id: $commodity,
                            year: $year,
                            tradeVolume: $volume,
                            dependencyScore: $dependency,
                            growth: $growth
                        }]->(t)
                        """,
                        source=source["iso3"], target=target["iso3"], commodity=commodity["id"],
                        year=year, volume=volume, dependency=dependency, growth=growth
                    )

async def main():
    await logger.ainfo("Connecting to Neo4j...")
    await neo4j_conn.connect()
    
    async with neo4j_conn.driver.session() as session:
        await logger.ainfo("Clearing existing graph...")
        await session.execute_write(clear_database)
        
        await logger.ainfo("Creating nodes...")
        await session.execute_write(create_nodes)
        
        await logger.ainfo("Creating relationships...")
        await session.execute_write(create_relationships)
        
    await neo4j_conn.close()
    await logger.ainfo("Successfully seeded Neo4j Trade Graph.")

if __name__ == "__main__":
    asyncio.run(main())
