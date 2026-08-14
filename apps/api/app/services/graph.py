from app.repositories.graph import GraphRepository
from app.schemas.graph import GraphResponse, CentralityResponse
from typing import List

class GraphService:
    def __init__(self):
        self.repo = GraphRepository()

    async def get_network(self) -> GraphResponse:
        return await self.repo.get_network()

    async def get_neighbors(self, node_id: str, depth: int) -> GraphResponse:
        return await self.repo.get_neighbors(node_id, depth)

    async def get_shortest_path(self, source_id: str, target_id: str) -> GraphResponse:
        return await self.repo.get_shortest_path(source_id, target_id)

    async def get_centrality(self) -> List[CentralityResponse]:
        records = await self.repo.get_centrality()
        return [CentralityResponse(node_id=r["node_id"], score=r["score"]) for r in records if r.get("node_id")]

graph_service = GraphService()
