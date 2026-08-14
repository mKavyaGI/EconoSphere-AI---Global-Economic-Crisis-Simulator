from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class GraphNode(BaseModel):
    id: str
    label: str
    properties: Dict[str, Any] = Field(default_factory=dict)

class GraphEdge(BaseModel):
    id: str
    source: str
    target: str
    type: str
    properties: Dict[str, Any] = Field(default_factory=dict)

class GraphResponse(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]

class ShortestPathRequest(BaseModel):
    source_id: str
    target_id: str
    relationship_type: Optional[str] = None
    max_depth: int = 5

class NeighborhoodRequest(BaseModel):
    node_id: str
    depth: int = 1
    relationship_type: Optional[str] = None

class CentralityResponse(BaseModel):
    node_id: str
    score: float
