from typing import Dict, Any, List
from app.repositories.graph import GraphRepository
from app.engine.interfaces import ShockVector

class DependencyResolver:
    """Queries Neo4j graph to find affected downstream nodes for propagation."""
    
    def __init__(self, graph_repo: GraphRepository):
        self.graph_repo = graph_repo

    async def resolve_propagation_paths(self, initial_shocks: ShockVector, max_depth: int = 1) -> Dict[str, Any]:
        """
        Given the initial shocked countries, query the graph to find their neighbors 
        up to `max_depth` hops. Returns a dependency map representing the subgraph.
        """
        dependency_map = {}
        for country_iso, impacts in initial_shocks.items():
            # In a real implementation we would batch this or use a custom Cypher query
            # For now, we reuse get_neighbors
            graph_data = await self.graph_repo.get_neighbors(node_id=country_iso, depth=max_depth)
            
            # Map edges to easily traverse from source -> targets
            country_dependencies = []
            for edge in graph_data.edges:
                if edge.source != edge.target:
                    # In a full setup we match Neo4j internal IDs back to ISO3,
                    # but for this MVP structure we assume edge source/target 
                    # corresponds to element IDs and we can map them back.
                    country_dependencies.append({
                        "source": edge.source,
                        "target": edge.target,
                        "type": edge.type,
                        "properties": edge.properties
                    })
            dependency_map[country_iso] = {
                "nodes": [n.model_dump() for n in graph_data.nodes],
                "edges": country_dependencies
            }
            
        return dependency_map
