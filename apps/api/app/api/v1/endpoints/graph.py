from fastapi import APIRouter, Depends, Query
from typing import List
from app.schemas.graph import GraphResponse, CentralityResponse
from app.services.graph import graph_service

router = APIRouter()

@router.get("/network", response_model=GraphResponse)
async def get_global_network():
    """Returns the global economic relationship graph."""
    return await graph_service.get_network()

@router.get("/neighbors/{node_id}", response_model=GraphResponse)
async def get_neighbors(node_id: str, depth: int = Query(1, ge=1, le=3)):
    """Returns the neighboring nodes and edges up to a specific depth."""
    return await graph_service.get_neighbors(node_id, depth)

@router.get("/path", response_model=GraphResponse)
async def get_shortest_path(source: str = Query(...), target: str = Query(...)):
    """Returns the shortest path between two nodes."""
    return await graph_service.get_shortest_path(source, target)

@router.get("/centrality", response_model=List[CentralityResponse])
async def get_centrality():
    """Returns nodes ranked by centrality."""
    return await graph_service.get_centrality()
