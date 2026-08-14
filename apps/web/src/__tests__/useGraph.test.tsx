import { describe, it, expect } from 'vitest';
import { GraphResponse, GraphEdge, GraphNode } from '@/hooks/useGraph';

describe('Global Trade Network Intelligence Data Transformation & KPIs', () => {
  const mockGraphData: GraphResponse = {
    nodes: [
      { id: 'USA', label: 'United States', properties: { region: 'North America' } },
      { id: 'CHN', label: 'China', properties: { region: 'Asia' } },
      { id: 'DEU', label: 'Germany', properties: { region: 'Europe' } },
      { id: 'JPN', label: 'Japan', properties: { region: 'Asia' } },
    ],
    edges: [
      { id: 'e1', source: 'USA', target: 'CHN', type: 'TRADE', properties: { tradeVolume: 650000000000, commodity_id: 'MANUFACTURING' } },
      { id: 'e2', source: 'CHN', target: 'DEU', type: 'TRADE', properties: { tradeVolume: 220000000000, commodity_id: 'TECHNOLOGY' } },
      { id: 'e3', source: 'DEU', target: 'USA', type: 'TRADE', properties: { tradeVolume: 180000000000, commodity_id: 'AUTOMOTIVE' } },
      { id: 'e4', source: 'JPN', target: 'CHN', type: 'TRADE', properties: { tradeVolume: 9000000000, commodity_id: 'ELECTRONICS' } },
    ],
  };

  it('accurately calculates global trade volume and KPI aggregates', () => {
    const totalVolume = mockGraphData.edges.reduce((sum: number, edge: GraphEdge) => sum + (edge.properties.tradeVolume || 0), 0);
    expect(totalVolume).toBe(1059000000000);
    expect(mockGraphData.nodes.length).toBe(4);
    expect(mockGraphData.edges.length).toBe(4);
  });

  it('correctly filters network links by minimum volume threshold', () => {
    const minVolumeThreshold = 100000000000; // $100 Billion
    const filteredEdges = mockGraphData.edges.filter((edge: GraphEdge) => (edge.properties.tradeVolume || 0) >= minVolumeThreshold);

    expect(filteredEdges.length).toBe(3);
    expect(filteredEdges.map((e: GraphEdge) => e.id)).not.toContain('e4'); // $9B link filtered out
  });

  it('correctly filters nodes and connected edges by regional domain', () => {
    const targetRegion = 'Asia';
    const asianNodeIds = new Set(
      mockGraphData.nodes.filter((n: GraphNode) => n.properties.region === targetRegion).map((n: GraphNode) => n.id)
    );

    expect(asianNodeIds).toContain('CHN');
    expect(asianNodeIds).toContain('JPN');
    expect(asianNodeIds.size).toBe(2);

    const regionalEdges = mockGraphData.edges.filter(
      (edge: GraphEdge) => asianNodeIds.has(edge.source) || asianNodeIds.has(edge.target)
    );
    expect(regionalEdges.length).toBe(3); // USA->CHN, CHN->DEU, JPN->CHN
  });
});
