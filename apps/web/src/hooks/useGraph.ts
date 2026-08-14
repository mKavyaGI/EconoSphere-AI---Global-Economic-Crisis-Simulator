import { useQuery } from '@tanstack/react-query';
import { apiClient } from '@/lib/api/client';

export interface GraphNodeProperties {
  name?: string;
  iso3?: string;
  region?: string;
  income_group?: string;
  health?: number;
  stability?: number;
  risk?: number;
  category?: string;
  type?: string;
  [key: string]: unknown;
}

export interface GraphNode {
  id: string;
  label: string;
  properties: GraphNodeProperties;
}

export interface GraphEdgeProperties {
  commodity_id?: string;
  year?: number;
  tradeVolume?: number;
  dependencyScore?: number;
  growth?: number;
  joined_year?: number;
  [key: string]: unknown;
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  type: string;
  properties: GraphEdgeProperties;
}

export interface GraphResponse {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface CentralityRank {
  node_id: string;
  score: number;
}

export function useGlobalNetwork() {
  return useQuery({
    queryKey: ['graph', 'network'],
    queryFn: async () => {
      const response = await apiClient.get<GraphResponse>('/graph/network');
      return response.data;
    },
    staleTime: 5 * 60 * 1000,
  });
}

export function useCentrality() {
  return useQuery({
    queryKey: ['graph', 'centrality'],
    queryFn: async () => {
      const response = await apiClient.get<CentralityRank[]>('/graph/centrality');
      return response.data;
    },
    staleTime: 5 * 60 * 1000,
  });
}

