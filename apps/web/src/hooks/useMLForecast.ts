/**
 * EconoSphere AI — Phase 11 Step 13
 * React Query hooks for the locked Production ML GDP Forecast.
 *
 * These hooks fetch from /api/v1/forecasts (the locked HistGradientBoostingRegressor pipeline),
 * NOT from /api/v1/ai/forecast (the generative AI system).
 *
 * The two systems are intentionally separate:
 *   - Production ML Forecast: deterministic, artifact-based, RMSE=3.9113, Q90=11.4507
 *   - AI Forecast: generative, multi-horizon, SHAP-explained (existing /ai/forecast endpoints)
 */
import { useQuery } from "@tanstack/react-query";
import { apiClient } from "@/lib/api/client";

// ─── Types ───────────────────────────────────────────────────────────────────

export interface ForecastModelInfo {
  version: string;
  algorithm: string;
  test_rmse: number;
  train_period: string;
  validation_period: string;
  test_period: string;
}

export interface ForecastUncertaintyInfo {
  method: string;
  method_description: string;
  q90: number;
  coverage_target: number;
  step12_decision: string;
}

export interface GDPForecastResponse {
  country_code: string;
  country_name: string;
  feature_year: number;
  forecast_year: number;
  predicted_gdp_growth: number;
  lower_bound_90: number;
  upper_bound_90: number;
  interval_width: number;
  forecast_status: "forecast";
  model: ForecastModelInfo;
  uncertainty: ForecastUncertaintyInfo;
}

export interface ForecastMetadata {
  model_version: string;
  algorithm: string;
  test_rmse: number;
  train_period: string;
  validation_period: string;
  test_period: string;
  feature_year: number;
  forecast_year: number;
  uncertainty_method: string;
  q90: number;
  coverage_target: number;
  step12_decision: string;
  artifact_file: string;
  available_countries: string[];
  n_countries: number;
}

export interface ForecastListResponse {
  forecasts: GDPForecastResponse[];
  metadata: ForecastMetadata;
  disclaimer: string;
}

// ─── Hooks ───────────────────────────────────────────────────────────────────

/**
 * Fetch all available country GDP growth forecasts from the locked ML pipeline.
 * Results are cached for 10 minutes (the artifact is static between ML runs).
 */
export function useMLForecasts() {
  return useQuery<ForecastListResponse>({
    queryKey: ["ml", "forecasts"],
    queryFn: async () => {
      const { data } = await apiClient.get<ForecastListResponse>("/forecasts");
      return data;
    },
    staleTime: 10 * 60 * 1000,
    retry: 2,
  });
}

/**
 * Fetch the locked ML GDP growth forecast for a specific country.
 * Returns undefined while loading, null if country has no forecast.
 */
export function useMLForecast(countryCode: string | undefined | null) {
  return useQuery<GDPForecastResponse>({
    queryKey: ["ml", "forecast", countryCode?.toUpperCase()],
    queryFn: async () => {
      const { data } = await apiClient.get<GDPForecastResponse>(
        `/forecasts/${countryCode!.toUpperCase()}`
      );
      return data;
    },
    enabled: !!countryCode && countryCode.trim().length === 3,
    staleTime: 10 * 60 * 1000,
    retry: (failureCount, error) => {
      // Don't retry 404 (country not in forecast set) or 400 (bad code)
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const status = (error as any)?.response?.status;
      if (status === 404 || status === 400) return false;
      return failureCount < 2;
    },
  });
}

/**
 * Fetch model provenance metadata for the locked ML pipeline.
 * Useful for displaying model information panels in the UI.
 */
export function useMLForecastMetadata() {
  return useQuery<ForecastMetadata>({
    queryKey: ["ml", "forecast", "metadata"],
    queryFn: async () => {
      const { data } = await apiClient.get<ForecastMetadata>("/forecasts/metadata");
      return data;
    },
    staleTime: 30 * 60 * 1000, // metadata almost never changes
    retry: 2,
  });
}
