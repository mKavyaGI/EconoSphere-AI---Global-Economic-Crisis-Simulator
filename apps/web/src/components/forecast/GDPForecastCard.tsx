"use client";

/**
 * EconoSphere AI — Phase 11 Step 13
 * GDPForecastCard — Single-country production ML forecast display.
 *
 * Displays the locked HistGradientBoostingRegressor forecast with:
 *  - Clear FORECAST label (never "ACTUAL")
 *  - Point estimate headline
 *  - 90% prediction interval visualization
 *  - Model provenance panel
 *  - Statistical disclaimer
 *
 * Props:
 *  forecast  — GDPForecastResponse from the locked ML pipeline
 *  compact   — if true, shows a condensed card without the model info panel
 */

import React from "react";
import { GDPForecastResponse } from "@/hooks/useMLForecast";
import {
  TrendingUp,
  TrendingDown,
  AlertCircle,
  Info,
  FlaskConical,
  Shield,
} from "lucide-react";

interface GDPForecastCardProps {
  forecast: GDPForecastResponse;
  compact?: boolean;
}

function formatPct(val: number): string {
  const sign = val >= 0 ? "+" : "";
  return `${sign}${val.toFixed(2)}%`;
}

/**
 * Interval bar: visualizes [lower, predicted, upper] on a horizontal axis.
 * The bar spans the full interval; the point marker sits at the predicted value.
 */
function IntervalBar({ lower, predicted, upper }: { lower: number; predicted: number; upper: number }) {
  const rangeMin = Math.min(lower, -15);
  const rangeMax = Math.max(upper, 15);
  const span = rangeMax - rangeMin;

  const lowerPct = ((lower - rangeMin) / span) * 100;
  const predictedPct = ((predicted - rangeMin) / span) * 100;
  const upperPct = ((upper - rangeMin) / span) * 100;
  const barWidth = upperPct - lowerPct;

  const zeroPct = ((0 - rangeMin) / span) * 100;

  return (
    <div className="mt-4 mb-2">
      <div className="flex justify-between text-[11px] font-mono text-slate-400 mb-1">
        <span>Lower (90%)</span>
        <span>Point Forecast</span>
        <span>Upper (90%)</span>
      </div>
      <div className="relative h-6 w-full">
        {/* Track */}
        <div className="absolute top-1/2 -translate-y-1/2 h-1.5 w-full bg-slate-700 rounded-full" />
        {/* Zero line */}
        <div
          className="absolute top-0 h-full w-px bg-slate-500"
          style={{ left: `${zeroPct}%` }}
        />
        {/* Interval band */}
        <div
          className="absolute top-1/2 -translate-y-1/2 h-3 rounded-full bg-blue-500/30 border border-blue-500/50"
          style={{ left: `${lowerPct}%`, width: `${barWidth}%` }}
        />
        {/* Point marker */}
        <div
          className="absolute top-1/2 -translate-y-1/2 w-4 h-4 rounded-full bg-blue-400 border-2 border-white shadow-lg shadow-blue-500/40 z-10"
          style={{ left: `${predictedPct}%`, transform: "translate(-50%, -50%)" }}
        />
      </div>
      {/* Labels */}
      <div className="relative mt-1" style={{ height: "16px" }}>
        <span
          className="absolute text-[10px] font-mono text-red-400 -translate-x-1/2"
          style={{ left: `${lowerPct}%` }}
        >
          {formatPct(lower)}
        </span>
        <span
          className="absolute text-[10px] font-mono text-blue-400 font-bold -translate-x-1/2"
          style={{ left: `${predictedPct}%` }}
        >
          {formatPct(predicted)}
        </span>
        <span
          className="absolute text-[10px] font-mono text-emerald-400 -translate-x-1/2"
          style={{ left: `${upperPct}%` }}
        >
          {formatPct(upper)}
        </span>
      </div>
    </div>
  );
}

export function GDPForecastCard({ forecast, compact = false }: GDPForecastCardProps) {
  const isPositive = forecast.predicted_gdp_growth >= 0;

  return (
    <div className="bg-slate-900 border border-blue-900/40 rounded-2xl overflow-hidden shadow-xl shadow-blue-900/10">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-950 via-slate-900 to-indigo-950 px-5 py-3 flex items-center justify-between border-b border-blue-900/30">
        <div className="flex items-center gap-2">
          <FlaskConical className="w-4 h-4 text-blue-400" />
          <span className="text-xs font-bold uppercase tracking-widest text-blue-300">
            Production ML Forecast
          </span>
        </div>
        <span className="px-2 py-0.5 text-[10px] font-extrabold uppercase tracking-wider rounded border border-amber-500/40 bg-amber-500/10 text-amber-300">
          FORECAST — MODEL GENERATED
        </span>
      </div>

      <div className="p-5 space-y-4">
        {/* Country + year */}
        <div className="flex items-start justify-between">
          <div>
            <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
              {forecast.country_name} · {forecast.country_code}
            </p>
            <p className="text-sm text-slate-400 mt-0.5">
              {forecast.forecast_year} GDP Growth Forecast
              <span className="ml-2 text-[10px] text-slate-500">
                (based on {forecast.feature_year} indicators)
              </span>
            </p>
          </div>
          {isPositive ? (
            <TrendingUp className="w-5 h-5 text-emerald-400 mt-0.5" />
          ) : (
            <TrendingDown className="w-5 h-5 text-red-400 mt-0.5" />
          )}
        </div>

        {/* Point forecast headline */}
        <div className="text-center py-3 border border-slate-800 rounded-xl bg-slate-950/50">
          <p className="text-xs text-slate-400 mb-1">Point Forecast</p>
          <p
            className={`text-4xl font-black font-mono ${
              isPositive ? "text-emerald-400" : "text-red-400"
            }`}
          >
            {formatPct(forecast.predicted_gdp_growth)}
          </p>
          <p className="text-xs text-slate-500 mt-1">percentage points</p>
        </div>

        {/* Interval bar */}
        <div className="bg-slate-950/40 rounded-xl p-4 border border-slate-800">
          <p className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-1 flex items-center gap-1">
            <Shield className="w-3.5 h-3.5 text-blue-400" />
            90% Prediction Range
          </p>
          <IntervalBar
            lower={forecast.lower_bound_90}
            predicted={forecast.predicted_gdp_growth}
            upper={forecast.upper_bound_90}
          />
          <div className="mt-3 grid grid-cols-3 gap-2 text-center">
            <div className="bg-slate-900 rounded-lg p-2">
              <p className="text-[10px] text-slate-500 mb-0.5">Lower</p>
              <p className="text-sm font-mono font-bold text-red-400">
                {formatPct(forecast.lower_bound_90)}
              </p>
            </div>
            <div className="bg-blue-950/40 rounded-lg p-2 border border-blue-900/30">
              <p className="text-[10px] text-blue-400 mb-0.5">Forecast</p>
              <p className="text-sm font-mono font-bold text-blue-300">
                {formatPct(forecast.predicted_gdp_growth)}
              </p>
            </div>
            <div className="bg-slate-900 rounded-lg p-2">
              <p className="text-[10px] text-slate-500 mb-0.5">Upper</p>
              <p className="text-sm font-mono font-bold text-emerald-400">
                {formatPct(forecast.upper_bound_90)}
              </p>
            </div>
          </div>
          <p className="text-center text-[10px] text-slate-500 mt-2">
            Interval width: {formatPct(forecast.interval_width)}
          </p>
        </div>

        {/* Model info panel — hidden in compact mode */}
        {!compact && (
          <div className="bg-slate-950/60 rounded-xl border border-slate-800 p-4">
            <p className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-3 flex items-center gap-1.5">
              <Info className="w-3.5 h-3.5 text-slate-400" />
              Model Information
            </p>
            <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-xs">
              <div>
                <p className="text-slate-500">Algorithm</p>
                <p className="text-slate-300 font-mono font-semibold">
                  HistGradientBoosting
                </p>
              </div>
              <div>
                <p className="text-slate-500">Test RMSE</p>
                <p className="text-emerald-400 font-mono font-bold">
                  {forecast.model.test_rmse.toFixed(4)} pp
                </p>
              </div>
              <div>
                <p className="text-slate-500">Training Period</p>
                <p className="text-slate-300 font-mono">{forecast.model.train_period}</p>
              </div>
              <div>
                <p className="text-slate-500">Validation</p>
                <p className="text-slate-300 font-mono">{forecast.model.validation_period}</p>
              </div>
              <div>
                <p className="text-slate-500">Test Period</p>
                <p className="text-slate-300 font-mono">{forecast.model.test_period}</p>
              </div>
              <div>
                <p className="text-slate-500">Uncertainty (Q90)</p>
                <p className="text-blue-400 font-mono font-bold">
                  ±{forecast.uncertainty.q90.toFixed(4)} pp
                </p>
              </div>
              <div className="col-span-2">
                <p className="text-slate-500">Uncertainty Method</p>
                <p className="text-slate-300 font-mono text-[11px]">
                  Global Q90 Residual Calibration
                </p>
              </div>
              <div className="col-span-2">
                <p className="text-slate-500">Step 12 Decision</p>
                <p className="text-amber-300 font-mono text-[11px]">
                  {forecast.uncertainty.step12_decision}
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Statistical disclaimer */}
        <div className="flex items-start gap-2 bg-amber-950/20 border border-amber-900/30 rounded-lg p-3">
          <AlertCircle className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
          <p className="text-[10px] text-amber-200/70 leading-relaxed">
            The point forecast is the model&apos;s single best estimate. The 90% prediction
            interval is calibrated from historical out-of-sample residuals and does{" "}
            <strong>not</strong> guarantee that actual GDP growth will fall within the stated
            range. This is a model forecast, not an observed economic statistic.
          </p>
        </div>
      </div>
    </div>
  );
}
