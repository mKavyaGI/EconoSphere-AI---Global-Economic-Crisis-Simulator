from neo4j import AsyncGraphDatabase, AsyncDriver
from app.core.config import settings

class Neo4jConnection:
    def __init__(self):
        self.driver: AsyncDriver | None = None

    async def connect(self):
        self.driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password)
        )
        await self.init_indexes()

    async def init_indexes(self):
        """Ensures Cypher performance indexes are present for fast O(1) graph traversal lookups."""
        if not self.driver:
            return
        try:
            async with self.driver.session() as session:
                await session.run("CREATE INDEX country_iso3_idx IF NOT EXISTS FOR (c:Country) ON (c.iso3)")
        except Exception:
            # Ignore offline database or authentication failures during test/local environments
            pass

    async def close(self):
        if self.driver is not None:
            await self.driver.close()

neo4j_conn = Neo4jConnection()
