/**
 * M-7: React Query hooks for AI Forecasting & Decision Intelligence (Phase 10).
 * Consistent with the rest of the project - all calls go through the shared apiClient,
 * which reads NEXT_PUBLIC_API_URL from environment variables.
 */
import { useQuery, useMutation } from '@tanstack/react-query';
import { apiClient } from '@/lib/api/client';

// ─── Types ───────────────────────────────────────────────────────────────────

export interface ConfidenceInterval {
  lower_bound_p10: number;
  median_p50: number;
  upper_bound_p90: number;
}

export interface ForecastPrediction {
  indicator_name: string;
  iso3: string;
  horizon: string;
  prediction_value: number;
  confidence_interval: ConfidenceInterval;
  confidence_score: number;
  timestamp: string;
  model_version: string;
  feature_importance: Record<string, number>;
  human_readable_explanation: string;
}

export interface CountryForecastResponse {
  iso3: string;
  generated_at: string;
  active_model_version: string;
  predictions: ForecastPrediction[];
  overall_economic_trajectory: string;
}

export interface RiskDimensionScore {
  dimension_name: string;
  score: number;
  tier: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  primary_contributors: string[];
}

export interface CountryRiskScorecard {
  iso3: string;
  timestamp: string;
  overall_risk_score: number;
  overall_risk_tier: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  dimensions: Record<string, RiskDimensionScore>;
}

export interface SimulatedTradeoff {
  indicator: string;
  horizon: string;
  delta_percentage: number;
  impact_direction: 'POSITIVE' | 'NEGATIVE' | 'NEUTRAL';
  rationale: string;
}

export interface PolicyRecommendation {
  recommendation_id: string;
  action_title: string;
  policy_type: string;
  target_iso3: string;
  reason: string;
  expected_benefit: string;
  confidence_score: number;
  risk_level: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  affected_countries: string[];
  affected_sectors: string[];
  simulated_tradeoffs: SimulatedTradeoff[];
}

export interface AnomalyAlert {
  anomaly_id: string;
  detected_anomaly: string;
  severity: 'LOW' | 'MODERATE' | 'SEVERE' | 'CRITICAL';
  timestamp: string;
  affected_countries: string[];
  possible_causes: string[];
  anomaly_score_z: number;
}

export interface ModelMetadata {
  model_id: string;
  model_type: string;
  version: string;
  trained_at: string;
  evaluation_metrics: Record<string, number>;
  deployment_status: string;
  hyperparameters: Record<string, unknown>;
  feature_list: string[];
}

// ─── Hooks ───────────────────────────────────────────────────────────────────

/**
 * Fetch multi-horizon AI macroeconomic forecast for a given country + model.
 * Results are cached by React Query for 5 minutes (staleTime).
 */
export function useForecast(iso3: string, modelType = 'XGBOOST', forceRefresh = false) {
  return useQuery<CountryForecastResponse>({
    queryKey: ['ai', 'forecast', iso3, modelType, forceRefresh],
    queryFn: async () => {
      const { data } = await apiClient.get<CountryForecastResponse>(
        `/ai/forecast/${iso3}`,
        { params: { model_type: modelType, force_refresh: forceRefresh } }
      );
      return data;
    },
    enabled: !!iso3,
    staleTime: 5 * 60 * 1000,
  });
}

/**
 * Fetch sovereign structural risk scorecard (7 dimensions, 0-100 score).
 */
export function useRisk(iso3: string, forceRefresh = false) {
  return useQuery<CountryRiskScorecard>({
    queryKey: ['ai', 'risk', iso3, forceRefresh],
    queryFn: async () => {
      const { data } = await apiClient.get<CountryRiskScorecard>(
        `/ai/risk/${iso3}`,
        { params: { force_refresh: forceRefresh } }
      );
      return data;
    },
    enabled: !!iso3,
    staleTime: 5 * 60 * 1000,
  });
}

/**
 * Fetch AI-generated prescriptive policy recommendations with simulated trade-offs.
 */
export function useRecommendations(iso3: string, simRunId?: string) {
  return useQuery<PolicyRecommendation[]>({
    queryKey: ['ai', 'recommendations', iso3, simRunId],
    queryFn: async () => {
      const { data } = await apiClient.get<PolicyRecommendation[]>(
        `/ai/recommendations/${iso3}`,
        { params: simRunId ? { sim_run_id: simRunId } : {} }
      );
      return data;
    },
    enabled: !!iso3,
    staleTime: 5 * 60 * 1000,
  });
}

/**
 * Fetch real-time global macroeconomic anomaly alerts.
 */
export function useAnomalies() {
  return useQuery<AnomalyAlert[]>({
    queryKey: ['ai', 'anomalies'],
    queryFn: async () => {
      const { data } = await apiClient.get<AnomalyAlert[]>('/ai/anomalies');
      return data;
    },
    staleTime: 2 * 60 * 1000,
    refetchInterval: 5 * 60 * 1000, // Auto-refresh every 5 minutes
  });
}

/**
 * Fetch all registered AI models with evaluation KPI metrics from the Model Registry.
 */
export function useAIModels() {
  return useQuery<ModelMetadata[]>({
    queryKey: ['ai', 'models'],
    queryFn: async () => {
      const { data } = await apiClient.get<ModelMetadata[]>('/ai/models');
      return data;
    },
    staleTime: 10 * 60 * 1000,
  });
}

/**
 * Dispatch an asynchronous background AI forecast job to the RQ worker queue.
 */
export function useDispatchAsyncForecast() {
  return useMutation({
    mutationFn: async ({ iso3, modelType }: { iso3: string; modelType?: string }) => {
      const { data } = await apiClient.post(
        `/ai/forecast/${iso3}/async`,
        {},
        { params: { model_type: modelType ?? 'XGBOOST' } }
      );
      return data;
    },
  });
}
