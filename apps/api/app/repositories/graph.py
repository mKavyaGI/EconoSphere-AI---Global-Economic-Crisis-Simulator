from typing import Dict, Any
from app.neo4j.session import neo4j_conn
from app.schemas.graph import GraphResponse, GraphNode, GraphEdge

class GraphRepository:
    async def run_query(self, query: str, parameters: Dict[str, Any] = None) -> GraphResponse:
        parameters = parameters or {}
        nodes_dict = {}
        edges_list = []
        
        async def fetch_data(tx):
            result = await tx.run(query, parameters)
            return [record async for record in result]
            
        async with neo4j_conn.driver.session() as session:
            records = await session.execute_read(fetch_data)
            
            for record in records:
                for key, value in record.items():
                    if hasattr(value, "labels"):  # It's a Node
                        node_id = str(value.element_id)
                        if node_id not in nodes_dict:
                            nodes_dict[node_id] = GraphNode(
                                id=node_id,
                                label=list(value.labels)[0] if value.labels else "Node",
                                properties=dict(value.items())
                            )
                    elif hasattr(value, "type"):  # It's a Relationship (Edge)
                        edge_id = str(value.element_id)
                        edges_list.append(GraphEdge(
                            id=edge_id,
                            source=str(value.start_node.element_id),
                            target=str(value.end_node.element_id),
                            type=value.type,
                            properties=dict(value.items())
                        ))
                    elif isinstance(value, list) and len(value) > 0 and hasattr(value[0], "nodes"): # Path
                        for path in value:
                            for node in path.nodes:
                                node_id = str(node.element_id)
                                if node_id not in nodes_dict:
                                    nodes_dict[node_id] = GraphNode(
                                        id=node_id,
                                        label=list(node.labels)[0] if node.labels else "Node",
                                        properties=dict(node.items())
                                    )
                            for edge in path.relationships:
                                edge_id = str(edge.element_id)
                                edges_list.append(GraphEdge(
                                    id=edge_id,
                                    source=str(edge.start_node.element_id),
                                    target=str(edge.end_node.element_id),
                                    type=edge.type,
                                    properties=dict(edge.items())
                                ))
                    elif hasattr(value, "nodes"): # Single Path
                        for node in value.nodes:
                            node_id = str(node.element_id)
                            if node_id not in nodes_dict:
                                nodes_dict[node_id] = GraphNode(
                                    id=node_id,
                                    label=list(node.labels)[0] if node.labels else "Node",
                                    properties=dict(node.items())
                                )
                        for edge in value.relationships:
                            edge_id = str(edge.element_id)
                            edges_list.append(GraphEdge(
                                id=edge_id,
                                source=str(edge.start_node.element_id),
                                target=str(edge.end_node.element_id),
                                type=edge.type,
                                properties=dict(edge.items())
                            ))
                            
        # Deduplicate edges just in case
        unique_edges = {e.id: e for e in edges_list}
        return GraphResponse(
            nodes=list(nodes_dict.values()),
            edges=list(unique_edges.values())
        )

    async def get_network(self) -> GraphResponse:
        query = """
        MATCH (n)-[r]->(m)
        RETURN n, r, m
        LIMIT 500
        """
        return await self.run_query(query)

    async def get_neighbors(self, node_id: str, depth: int = 1) -> GraphResponse:
        # Assumes node_id is the primary key (iso3 or similar identifier) passed as parameter
        # We search across any label for an identifier property `iso3` or `id`.
        query = f"""
        MATCH p=(n)-[*1..{depth}]-(m)
        WHERE n.iso3 = $node_id OR n.id = $node_id
        RETURN p
        LIMIT 200
        """
        return await self.run_query(query, {"node_id": node_id})

    async def get_shortest_path(self, source_id: str, target_id: str) -> GraphResponse:
        query = """
        MATCH p=shortestPath((n)-[*]-(m))
        WHERE (n.iso3 = $source_id OR n.id = $source_id) AND (m.iso3 = $target_id OR m.id = $target_id)
        RETURN p
        """
        return await self.run_query(query, {"source_id": source_id, "target_id": target_id})
    
    async def get_centrality(self):
        # Returns out-degree as a centrality proxy (avoids requiring GDS plugin).
        query = """
        MATCH (n)
        OPTIONAL MATCH (n)-[r]->()
        RETURN n.iso3 AS node_id, count(r) AS score
        ORDER BY score DESC
        LIMIT 10
        """
        async def fetch(tx):
            result = await tx.run(query)
            return await result.data()

        async with neo4j_conn.driver.session() as session:
            return await session.execute_read(fetch)
