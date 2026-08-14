import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '@/lib/api/client';

export interface ScenarioEvent {
  id: number;
  version_id: number;
  parent_event_id: number | null;
  title: string;
  category: string;
  status: string;
  parameters: Record<string, unknown>;
  shockIntensity?: number;
  propagationDelay?: number;
  recoveryRate?: number;
  affectedNodes?: string[];
  confidence?: number;
  simulationWeight?: number;
}

export interface ScenarioVersion {
  id: number;
  scenario_id: number;
  version_number: number;
  created_at: string;
  events: ScenarioEvent[];
}

export interface Scenario {
  id: number;
  title: string;
  description: string;
  author: string | null;
  status?: string;
  created_at: string;
  updated_at: string | null;
  is_template: boolean;
  tags: string[];
  versions: ScenarioVersion[];
}

export interface ScenarioValidationResult {
  is_valid: boolean;
  errors: string[];
  warnings: string[];
}

export interface UseScenariosParams {
  isTemplate?: boolean;
  status?: string;
  title?: string;
  skip?: number;
  limit?: number;
}

export function useScenarios(paramsOrBool: UseScenariosParams | boolean = {}) {
  const params: UseScenariosParams = typeof paramsOrBool === 'boolean' ? { isTemplate: paramsOrBool } : paramsOrBool;
  return useQuery({
    queryKey: ['scenarios', params],
    queryFn: async () => {
      const queryParams = new URLSearchParams();
      if (params.isTemplate !== undefined) queryParams.append('is_template', String(params.isTemplate));
      if (params.status) queryParams.append('status', params.status);
      if (params.title) queryParams.append('title', params.title);
      if (params.skip !== undefined) queryParams.append('skip', String(params.skip));
      if (params.limit !== undefined) queryParams.append('limit', String(params.limit));
      
      const queryString = queryParams.toString();
      const url = `/scenarios${queryString ? `?${queryString}` : ''}`;
      const response = await apiClient.get<Scenario[]>(url);
      return response.data;
    },
    staleTime: 60 * 1000,
  });
}

export function useScenario(id: number) {
  return useQuery({
    queryKey: ['scenarios', id],
    queryFn: async () => {
      const response = await apiClient.get<Scenario>(`/scenarios/${id}`);
      return response.data;
    },
    enabled: !!id,
  });
}

export function useCreateScenario() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (data: Partial<Scenario>) => {
      const response = await apiClient.post<Scenario>('/scenarios', data);
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['scenarios'] });
    },
  });
}

export function useUpdateScenario() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async ({ id, data }: { id: number; data: Partial<Scenario> }) => {
      const response = await apiClient.put<Scenario>(`/scenarios/${id}`, data);
      return response.data;
    },
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ['scenarios'] });
      queryClient.invalidateQueries({ queryKey: ['scenarios', variables.id] });
    },
  });
}

export function useDeleteScenario() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (id: number) => {
      await apiClient.delete(`/scenarios/${id}`);
      return id;
    },
    onSuccess: (id) => {
      queryClient.invalidateQueries({ queryKey: ['scenarios'] });
      queryClient.removeQueries({ queryKey: ['scenarios', id] });
    },
  });
}

export function useDuplicateScenario() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async ({ id, newTitle }: { id: number; newTitle: string }) => {
      const response = await apiClient.post<Scenario>(`/scenarios/${id}/duplicate?new_title=${encodeURIComponent(newTitle)}`);
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['scenarios'] });
    },
  });
}

export function usePublishScenario() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (id: number) => {
      const response = await apiClient.post<Scenario>(`/scenarios/${id}/publish`);
      return response.data;
    },
    onSuccess: (_, id) => {
      queryClient.invalidateQueries({ queryKey: ['scenarios'] });
      queryClient.invalidateQueries({ queryKey: ['scenarios', id] });
    },
  });
}

export function useArchiveScenario() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (id: number) => {
      const response = await apiClient.post<Scenario>(`/scenarios/${id}/archive`);
      return response.data;
    },
    onSuccess: (_, id) => {
      queryClient.invalidateQueries({ queryKey: ['scenarios'] });
      queryClient.invalidateQueries({ queryKey: ['scenarios', id] });
    },
  });
}

export function useValidateScenario() {
  return useMutation({
    mutationFn: async (id: number) => {
      const response = await apiClient.post<ScenarioValidationResult>(`/scenarios/${id}/validate`);
      return response.data;
    },
  });
}

export function useCreateEvent() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async ({ scenarioId, data }: { scenarioId: number; data: Partial<ScenarioEvent> }) => {
      const response = await apiClient.post<ScenarioEvent>(`/scenarios/${scenarioId}/events/`, data);
      return response.data;
    },
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ['scenarios', variables.scenarioId] });
      queryClient.invalidateQueries({ queryKey: ['scenarios'] });
    },
  });
}

export function useUpdateEvent() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async ({ scenarioId, eventId, data }: { scenarioId: number; eventId: number; data: Partial<ScenarioEvent> }) => {
      const response = await apiClient.put<ScenarioEvent>(`/scenarios/${scenarioId}/events/${eventId}`, data);
      return response.data;
    },
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ['scenarios', variables.scenarioId] });
      queryClient.invalidateQueries({ queryKey: ['scenarios'] });
    },
  });
}

export function useDeleteEvent() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async ({ scenarioId, eventId }: { scenarioId: number; eventId: number }) => {
      await apiClient.delete(`/scenarios/${scenarioId}/events/${eventId}`);
      return eventId;
    },
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ['scenarios', variables.scenarioId] });
      queryClient.invalidateQueries({ queryKey: ['scenarios'] });
    },
  });
}

