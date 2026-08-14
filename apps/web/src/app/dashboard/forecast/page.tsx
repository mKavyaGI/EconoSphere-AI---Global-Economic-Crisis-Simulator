"use client";

import React, { useState, useEffect } from "react";
import ReactECharts from "echarts-for-react";
import {
  Brain,
  TrendingUp,
  AlertTriangle,
  CheckCircle,
  HelpCircle,
  RefreshCw,
  Globe,
  ShieldAlert,
  ArrowUpRight,
  ArrowDownRight,
  Activity,
  Layers,
  BarChart3,
  Cpu
} from "lucide-react";
import {
  useForecast,
  useRisk,
  useRecommendations,
  useAnomalies,
  type ForecastPrediction,
} from "@/hooks/useAI";

const SUPPORTED_COUNTRIES = ["USA", "CHN", "DEU", "JPN", "IND", "GBR", "BRA", "FRA", "CAN", "AUS"];
const SUPPORTED_MODELS = ["XGBOOST", "TRANSFORMER", "GNN", "LIGHTGBM", "VAR", "RL"];
const INDICATORS = [
  "GDP Growth",
  "Inflation Rate",
  "Unemployment Rate",
  "Interest Rate",
  "Exchange Rate",
  "Government Debt-to-GDP",
  "Trade Volume",
  "Currency Strength Index"
];

export default function AIForecastDashboard() {
  const [selectedIso, setSelectedIso] = useState<string>("USA");
  const [selectedModel, setSelectedModel] = useState<string>("XGBOOST");
  const [selectedIndicator, setSelectedIndicator] = useState<string>("GDP Growth");
  const [forceRefresh, setForceRefresh] = useState<boolean>(false);
  const [selectedExplainPrediction, setSelectedExplainPrediction] = useState<ForecastPrediction | null>(null);

  // M-1 Fix: Use React Query hooks (respects NEXT_PUBLIC_API_URL env var via shared apiClient)
  const { data: forecastData, isLoading: forecastLoading, error: forecastError, refetch: refetchForecast } =
    useForecast(selectedIso, selectedModel, forceRefresh);
  const { data: riskScorecard, isLoading: riskLoading, refetch: refetchRisk } =
    useRisk(selectedIso, forceRefresh);
  const { data: recommendations = [], isLoading: recLoading, refetch: refetchRec } =
    useRecommendations(selectedIso);
  const { data: anomalies = [], refetch: refetchAnomalies } =
    useAnomalies();

  const loading = forecastLoading || riskLoading || recLoading;
  const error = forecastError ? (forecastError as Error).message : null;

  const predictions = React.useMemo(() => forecastData?.predictions ?? [], [forecastData?.predictions]);
  const trajectory = forecastData?.overall_economic_trajectory ?? "STABILIZING_MODERATE";
  const activeVersion = forecastData?.active_model_version ?? "1.0.0-PROD";

  const handleRefresh = () => {
    setForceRefresh(true);
    refetchForecast();
    refetchRisk();
    refetchRec();
    refetchAnomalies();
    // Reset flag after a tick so React Query doesn't permanently cache with force_refresh=true
    setTimeout(() => setForceRefresh(false), 500);
  };

  useEffect(() => {
    if (predictions.length > 0) {
      const match =
        predictions.find((p: ForecastPrediction) => p.indicator_name === selectedIndicator && p.horizon === "1_year") ??
        predictions.find((p: ForecastPrediction) => p.indicator_name === selectedIndicator);
      if (match) {
        const id = requestAnimationFrame(() => setSelectedExplainPrediction(match));
        return () => cancelAnimationFrame(id);
      }
    }
  }, [selectedIndicator, predictions]);

  // Filter series for selected indicator across horizons
  const indicatorSeries = predictions.filter(p => p.indicator_name === selectedIndicator);
  
  // Format ECharts option for Confidence Bands
  const getChartOption = () => {
    const horizonLabels = ["3 Months", "6 Months", "1 Year", "3 Years", "5 Years"];
    const horizonKeys = ["3_months", "6_months", "1_year", "3_years", "5_years"];
    
    const p10 = horizonKeys.map(h => {
      const item = indicatorSeries.find(p => p.horizon === h);
      return item ? item.confidence_interval.lower_bound_p10 : 0;
    });
    const p50 = horizonKeys.map(h => {
      const item = indicatorSeries.find(p => p.horizon === h);
      return item ? item.prediction_value : 0;
    });
    const p90 = horizonKeys.map(h => {
      const item = indicatorSeries.find(p => p.horizon === h);
      return item ? item.confidence_interval.upper_bound_p90 : 0;
    });

    return {
      backgroundColor: "transparent",
      tooltip: {
        trigger: "axis",
        backgroundColor: "#0f172a",
        borderColor: "#334155",
        textStyle: { color: "#f8fafc" },
        formatter: (params: Array<{ axisValue: string; seriesName: string; value: number | string }>) => {
          let str = `<div class="font-bold border-b border-slate-700 pb-1 mb-1">${params[0].axisValue} Projection</div>`;
          params.forEach(p => {
            const color = p.seriesName === "Median Expectation (P50)" ? "#38bdf8" : "#94a3b8";
            str += `<div class="flex justify-between items-center my-1"><span style="color:${color}; font-weight:600;">${p.seriesName}:</span> <span class="ml-4 font-mono">${p.value}</span></div>`;
          });
          return str;
        }
      },
      legend: {
        data: ["Upper Bound (P90)", "Median Expectation (P50)", "Lower Bound (P10)"],
        textStyle: { color: "#94a3b8" },
        bottom: 0
      },
      grid: { left: "4%", right: "4%", top: "10%", bottom: "15%", containLabel: true },
      xAxis: {
        type: "category",
        boundaryGap: false,
        data: horizonLabels,
        axisLine: { lineStyle: { color: "#475569" } },
        axisLabel: { color: "#cbd5e1", fontWeight: 600 }
      },
      yAxis: {
        type: "value",
        splitLine: { lineStyle: { color: "rgba(51, 65, 85, 0.4)", type: "dashed" } },
        axisLabel: { color: "#cbd5e1" }
      },
      series: [
        {
          name: "Upper Bound (P90)",
          type: "line",
          data: p90,
          lineStyle: { opacity: 0 },
          stack: "confidence-band",
          symbol: "none"
        },
        {
          name: "Lower Bound (P10)",
          type: "line",
          data: p10,
          lineStyle: { opacity: 0 },
          areaStyle: {
            color: {
              type: "linear",
              x: 0, y: 0, x2: 0, y2: 1,
              colorStops: [
                { offset: 0, color: "rgba(56, 189, 248, 0.3)" },
                { offset: 1, color: "rgba(56, 189, 248, 0.05)" }
              ]
            }
          },
          stack: "confidence-band",
          symbol: "none"
        },
        {
          name: "Median Expectation (P50)",
          type: "line",
          data: p50,
          smooth: true,
          lineStyle: { color: "#38bdf8", width: 4 },
          itemStyle: { color: "#0284c7", borderColor: "#38bdf8", borderWidth: 2 },
          symbolSize: 8
        }
      ]
    };
  };

  // Format ECharts option for SHAP Feature Importance
  const getShapChartOption = (impMap: Record<string, number>) => {
    const entries = Object.entries(impMap || {}).sort((a, b) => b[1] - a[1]);
    const labels = entries.map(e => e[0].replace(/_/g, " ").replace(/\b\w/g, l => l.toUpperCase()));
    const values = entries.map(e => e[1]);

    return {
      backgroundColor: "transparent",
      tooltip: {
        trigger: "axis",
        axisPointer: { type: "shadow" },
        formatter: "{b}: <b>{c}%</b> relative contribution"
      },
      grid: { left: "3%", right: "8%", top: "5%", bottom: "5%", containLabel: true },
      xAxis: {
        type: "value",
        axisLabel: { formatter: "{value}%", color: "#94a3b8" },
        splitLine: { lineStyle: { color: "rgba(51, 65, 85, 0.3)" } }
      },
      yAxis: {
        type: "category",
        data: labels.reverse(),
        axisLabel: { color: "#cbd5e1", fontSize: 11 },
        axisLine: { lineStyle: { color: "#475569" } }
      },
      series: [
        {
          type: "bar",
          data: values.reverse(),
          itemStyle: {
            color: {
              type: "linear",
              x: 0, y: 0, x2: 1, y2: 0,
              colorStops: [
                { offset: 0, color: "#3b82f6" },
                { offset: 1, color: "#06b6d4" }
              ]
            },
            borderRadius: [0, 4, 4, 0]
          },
          barWidth: "60%"
        }
      ]
    };
  };

  const getRiskColor = (tier?: string) => {
    switch (tier) {
      case "LOW": return "text-emerald-400 border-emerald-500/30 bg-emerald-500/10";
      case "MEDIUM": return "text-amber-400 border-amber-500/30 bg-amber-500/10";
      case "HIGH": return "text-orange-400 border-orange-500/30 bg-orange-500/10";
      case "CRITICAL": return "text-rose-400 border-rose-500/30 bg-rose-500/10";
      default: return "text-slate-400 border-slate-700 bg-slate-800";
    }
  };

  return (
    <div className="min-h-screen bg-[#090d16] text-slate-100 p-6 md:p-8 font-sans">
      {/* Header & Global Model Controls */}
      <header className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 mb-8 pb-6 border-b border-slate-800">
        <div>
          <div className="flex items-center gap-3">
            <div className="p-2.5 bg-gradient-to-tr from-blue-600 to-cyan-500 rounded-xl shadow-lg shadow-cyan-500/20">
              <Brain className="w-7 h-7 text-white animate-pulse" />
            </div>
            <div>
              <h1 className="text-2xl md:text-3xl font-extrabold tracking-tight bg-gradient-to-r from-white via-slate-200 to-cyan-400 bg-clip-text text-transparent">
                AI Forecasting & Decision Intelligence
              </h1>
              <p className="text-sm text-slate-400">
                Multi-horizon economic projections, sovereign risk scorecards, and prescriptive policy trade-offs.
              </p>
            </div>
          </div>
        </div>

        {/* Global Controls */}
        <div className="flex flex-wrap items-center gap-3 w-full md:w-auto">
          <div className="flex items-center gap-2 bg-slate-900/90 border border-slate-800 rounded-lg px-3 py-1.5 shadow-inner">
            <Globe className="w-4 h-4 text-cyan-400" />
            <span className="text-xs font-semibold text-slate-400 uppercase">Country:</span>
            <select
              value={selectedIso}
              onChange={(e) => setSelectedIso(e.target.value)}
              className="bg-transparent text-sm font-bold text-cyan-300 focus:outline-none cursor-pointer"
            >
              {SUPPORTED_COUNTRIES.map(iso => (
                <option key={iso} value={iso} className="bg-slate-900 text-white">{iso}</option>
              ))}
            </select>
          </div>

          <div className="flex items-center gap-2 bg-slate-900/90 border border-slate-800 rounded-lg px-3 py-1.5 shadow-inner">
            <Cpu className="w-4 h-4 text-purple-400" />
            <span className="text-xs font-semibold text-slate-400 uppercase">Model:</span>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="bg-transparent text-sm font-bold text-purple-300 focus:outline-none cursor-pointer"
            >
              {SUPPORTED_MODELS.map(mod => (
                <option key={mod} value={mod} className="bg-slate-900 text-white">{mod}</option>
              ))}
            </select>
          </div>

          <button
            onClick={handleRefresh}
            disabled={loading}
            className="flex items-center gap-2 px-4 py-2 bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 text-white text-sm font-semibold rounded-lg shadow-lg shadow-cyan-600/30 transition-all active:scale-95 cursor-pointer"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? "animate-spin" : ""}`} />
            <span>Refresh AI</span>
          </button>
        </div>
      </header>

      {error && (
        <div className="mb-6 p-4 bg-rose-950/50 border border-rose-800/80 rounded-xl text-rose-300 flex items-center gap-3">
          <AlertTriangle className="w-6 h-6 text-rose-500 flex-shrink-0" />
          <div>
            <p className="font-bold">AI Engine Communication Warning</p>
            <p className="text-sm text-rose-400/90">{error} — Check that the backend API is running.</p>
          </div>
        </div>
      )}

      {/* Top Banner: Active Model Profile & Trajectory Tag */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
        <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4 flex items-center justify-between">
          <div>
            <p className="text-xs text-slate-400 font-semibold uppercase tracking-wider">Macro Trajectory</p>
            <p className="text-lg font-black mt-1 text-white flex items-center gap-2">
              {trajectory.replace("_", " ")}
              <Activity className="w-4 h-4 text-emerald-400" />
            </p>
          </div>
        </div>

        <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4 flex items-center justify-between">
          <div>
            <p className="text-xs text-slate-400 font-semibold uppercase tracking-wider">Active ML Architecture</p>
            <p className="text-lg font-black mt-1 text-purple-400">{selectedModel} <span className="text-xs font-normal text-slate-500">v{activeVersion}</span></p>
          </div>
          <Layers className="w-8 h-8 text-purple-500/40" />
        </div>

        <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4 flex items-center justify-between">
          <div>
            <p className="text-xs text-slate-400 font-semibold uppercase tracking-wider">Overall Risk Score</p>
            <div className="flex items-center gap-2 mt-1">
              <span className="text-2xl font-black text-white">{riskScorecard?.overall_risk_score || 38.4}/100</span>
              <span className={`px-2 py-0.5 text-xs font-bold rounded border ${getRiskColor(riskScorecard?.overall_risk_tier || "MEDIUM")}`}>
                {riskScorecard?.overall_risk_tier || "MEDIUM"}
              </span>
            </div>
          </div>
          <ShieldAlert className="w-8 h-8 text-amber-500/40" />
        </div>

        <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4 flex items-center justify-between">
          <div>
            <p className="text-xs text-slate-400 font-semibold uppercase tracking-wider">Systemic Anomalies</p>
            <p className="text-2xl font-black mt-1 text-cyan-400">{anomalies.length} <span className="text-xs font-normal text-slate-400">active alerts</span></p>
          </div>
          <BarChart3 className="w-8 h-8 text-cyan-500/40" />
        </div>
      </div>

      {/* Main Grid: Forecasting Chart & Explainability Drawer */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 mb-8">
        {/* Chart Panel (8 cols) */}
        <div className="lg:col-span-8 bg-slate-900/80 border border-slate-800/90 rounded-2xl p-6 shadow-2xl backdrop-blur-md">
          <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 mb-6">
            <div>
              <h2 className="text-xl font-bold text-white flex items-center gap-2">
                <TrendingUp className="w-5 h-5 text-cyan-400" />
                <span>Multi-Horizon Projection: {selectedIndicator}</span>
              </h2>
              <p className="text-xs text-slate-400 mt-0.5">
                Displays point predictions (P50) enclosed by 10th (P10) and 90th (P90) percentile uncertainty boundaries.
              </p>
            </div>

            {/* Indicator Switcher Pill */}
            <div className="w-full sm:w-auto overflow-x-auto pb-1">
              <select
                value={selectedIndicator}
                onChange={(e) => setSelectedIndicator(e.target.value)}
                className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm font-semibold text-cyan-300 focus:outline-none cursor-pointer shadow"
              >
                {INDICATORS.map(ind => (
                  <option key={ind} value={ind}>{ind}</option>
                ))}
              </select>
            </div>
          </div>

          {/* Chart Viewport */}
          <div className="h-[340px] w-full pt-2">
            {loading ? (
              <div className="w-full h-full flex flex-col items-center justify-center text-slate-500">
                <RefreshCw className="w-8 h-8 animate-spin text-cyan-500 mb-2" />
                <span>Computing deep predictive sequence models...</span>
              </div>
            ) : indicatorSeries.length > 0 ? (
              <ReactECharts option={getChartOption()} style={{ height: "100%", width: "100%" }} notMerge={true} />
            ) : (
              <div className="w-full h-full flex items-center justify-center text-slate-500">
                No forecast series available for {selectedIndicator}. Try clicking Refresh AI.
              </div>
            )}
          </div>
        </div>

        {/* Explainable AI (XAI) Panel (4 cols) */}
        <div className="lg:col-span-4 bg-slate-900/80 border border-slate-800/90 rounded-2xl p-6 shadow-2xl flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-4 pb-3 border-b border-slate-800">
              <h3 className="text-base font-bold text-white flex items-center gap-2">
                <HelpCircle className="w-5 h-5 text-purple-400" />
                <span>Explainable AI (XAI) Diagnosis</span>
              </h3>
              <span className="text-[10px] uppercase tracking-widest bg-purple-500/20 text-purple-300 border border-purple-500/30 px-2 py-0.5 rounded font-mono font-bold">
                TreeSHAP / NLG
              </span>
            </div>

            {selectedExplainPrediction ? (
              <div className="space-y-4">
                {/* NLG Box */}
                <div className="p-4 bg-gradient-to-r from-slate-950 via-slate-900 to-slate-950 border border-purple-900/40 rounded-xl shadow-inner">
                  <p className="text-xs font-semibold text-purple-300 mb-1 uppercase tracking-wider">Executive Synthesis</p>
                  <p className="text-sm leading-relaxed text-slate-300 italic">
                    &ldquo;{selectedExplainPrediction.human_readable_explanation}&rdquo;
                  </p>
                </div>

                {/* SHAP Contribution Chart */}
                <div>
                  <p className="text-xs font-bold uppercase text-slate-400 mb-2 tracking-wider">Top Driver Attributions (SHAP %)</p>
                  <div className="h-[180px] w-full">
                    <ReactECharts
                      option={getShapChartOption(selectedExplainPrediction.feature_importance)}
                      style={{ height: "100%", width: "100%" }}
                      notMerge={true}
                    />
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-sm text-slate-500 py-12 text-center">
                Select a valid indicator series to view SHAP causal attributions.
              </div>
            )}
          </div>

          <div className="mt-4 pt-3 border-t border-slate-800/80 flex items-center justify-between text-xs text-slate-400 font-mono">
            <span>Model Confidence Meter:</span>
            <span className="text-emerald-400 font-bold text-sm">
              {((selectedExplainPrediction?.confidence_score || 0.94) * 100).toFixed(1)}% Certainty
            </span>
          </div>
        </div>
      </div>

      {/* Sovereign Risk Scorecard Heatmap */}
      <div className="mb-10">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-xl font-bold text-white flex items-center gap-2">
              <ShieldAlert className="w-5 h-5 text-amber-400" />
              <span>Sovereign Risk Assessment Heatmap (7 Dimensions)</span>
            </h2>
            <p className="text-xs text-slate-400">
              Normalized scorecards from 0.0 (safest) to 100.0 (extreme economic hazard) across structural pillars.
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {riskScorecard && Object.entries(riskScorecard.dimensions).map(([key, dim]) => (
            <div key={key} className="bg-slate-900/70 border border-slate-800 rounded-xl p-4 hover:border-slate-700 transition-all">
              <div className="flex justify-between items-start">
                <span className="text-sm font-bold text-slate-200">{dim.dimension_name}</span>
                <span className={`px-2 py-0.5 text-[10px] font-extrabold rounded border ${getRiskColor(dim.tier)}`}>
                  {dim.tier}
                </span>
              </div>
              <div className="mt-3 flex items-baseline gap-1">
                <span className="text-2xl font-black text-white">{dim.score}</span>
                <span className="text-xs text-slate-500 font-mono">/ 100</span>
              </div>
              {/* Progress Bar */}
              <div className="w-full bg-slate-800 h-1.5 rounded-full mt-2 overflow-hidden">
                <div
                  className={`h-full rounded-full ${dim.score > 70 ? "bg-rose-500" : dim.score > 50 ? "bg-amber-400" : "bg-emerald-400"}`}
                  style={{ width: `${dim.score}%` }}
                />
              </div>
              <div className="mt-3 text-[11px] text-slate-400 font-mono">
                <span className="text-slate-500">Drivers:</span> {dim.primary_contributors.join(", ")}
              </div>
            </div>
          ))}
          {!riskScorecard && (
            <div className="col-span-4 p-6 bg-slate-900/40 rounded-xl text-center text-slate-500 border border-slate-800 border-dashed">
              Loading sovereign risk dimensions...
            </div>
          )}
        </div>
      </div>

      {/* Prescriptive Policy Recommendations & Simulated Tradeoffs */}
      <div>
        <div className="mb-4">
          <h2 className="text-xl font-bold text-white flex items-center gap-2">
            <CheckCircle className="w-5 h-5 text-emerald-400" />
            <span>AI Actionable Policy Recommendations & Simulated Trade-offs</span>
          </h2>
          <p className="text-xs text-slate-400">
            Prescriptive decision support directives paired with explicit secondary tradeoff consequences computed via simulation engines.
          </p>
        </div>

        <div className="grid grid-cols-1 gap-6">
          {recommendations.map((rec) => (
            <div key={rec.recommendation_id} className="bg-slate-900/90 border border-slate-800 rounded-2xl p-6 shadow-xl hover:border-slate-700/80 transition-all">
              <div className="flex flex-col lg:flex-row justify-between items-start lg:items-center gap-4 pb-4 border-b border-slate-800">
                <div>
                  <div className="flex items-center gap-2 mb-1">
                    <span className="px-2 py-0.5 bg-cyan-950/80 border border-cyan-700 text-cyan-300 font-mono text-[11px] font-bold rounded">
                      {rec.policy_type}
                    </span>
                    <span className="text-xs font-mono text-slate-500">{rec.recommendation_id}</span>
                  </div>
                  <h3 className="text-lg font-bold text-white">{rec.action_title}</h3>
                </div>

                <div className="flex items-center gap-4">
                  <div className="text-right">
                    <p className="text-[11px] text-slate-400 uppercase">AI Confidence</p>
                    <p className="text-sm font-black text-emerald-400">{(rec.confidence_score * 100).toFixed(0)}% Efficacy</p>
                  </div>
                  <div className="text-right">
                    <p className="text-[11px] text-slate-400 uppercase">Implementation Risk</p>
                    <span className={`px-2 py-0.5 text-xs font-extrabold rounded border ${getRiskColor(rec.risk_level)}`}>
                      {rec.risk_level}
                    </span>
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 my-4">
                <div className="bg-slate-950/50 p-4 rounded-xl border border-slate-800/80">
                  <p className="text-xs font-bold uppercase text-slate-400 mb-1">Diagnostic Trigger Reason</p>
                  <p className="text-sm text-slate-300">{rec.reason}</p>
                </div>
                <div className="bg-emerald-950/20 p-4 rounded-xl border border-emerald-800/30">
                  <p className="text-xs font-bold uppercase text-emerald-400 mb-1">Quantified Expected Benefit</p>
                  <p className="text-sm text-emerald-200/90 font-semibold">{rec.expected_benefit}</p>
                </div>
              </div>

              {/* Affected Entities Pills */}
              <div className="flex flex-wrap items-center gap-2 mb-4 text-xs">
                <span className="text-slate-400 font-semibold">Affected Partners:</span>
                {rec.affected_countries.map(c => (
                  <span key={c} className="px-2 py-0.5 bg-slate-800 text-cyan-300 rounded-md font-mono font-bold">{c}</span>
                ))}
                <span className="text-slate-400 font-semibold ml-4">Ripple Sectors:</span>
                {rec.affected_sectors.map(s => (
                  <span key={s} className="px-2 py-0.5 bg-slate-800 text-purple-300 rounded-md">{s}</span>
                ))}
              </div>

              {/* Simulated Trade-off Matrix */}
              {rec.simulated_tradeoffs && rec.simulated_tradeoffs.length > 0 && (
                <div className="mt-4 pt-4 border-t border-slate-800/80">
                  <p className="text-xs font-bold uppercase tracking-wider text-amber-400/90 mb-3 flex items-center gap-1.5">
                    <AlertTriangle className="w-4 h-4 text-amber-400" />
                    <span>Simulated Secondary KPI Trade-offs</span>
                  </p>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    {rec.simulated_tradeoffs.map((to, idx) => (
                      <div key={idx} className="bg-slate-950/80 border border-slate-800 rounded-lg p-3 flex flex-col justify-between">
                        <div className="flex items-center justify-between font-bold text-sm mb-1">
                          <span className="text-slate-200">{to.indicator} <span className="text-xs font-normal text-slate-500">({to.horizon.replace("_", " ")})</span></span>
                          <span className={to.delta_percentage >= 0 ? "text-emerald-400 flex items-center" : "text-rose-400 flex items-center"}>
                            {to.delta_percentage > 0 ? <ArrowUpRight className="w-4 h-4 mr-0.5" /> : <ArrowDownRight className="w-4 h-4 mr-0.5" />}
                            {to.delta_percentage > 0 ? `+${to.delta_percentage}%` : `${to.delta_percentage}%`}
                          </span>
                        </div>
                        <p className="text-xs text-slate-400 italic">{to.rationale}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ))}

          {recommendations.length === 0 && !loading && (
            <div className="p-8 bg-slate-900/40 rounded-2xl text-center text-slate-500 border border-slate-800 border-dashed">
              No active strategic policy recommendations generated for {selectedIso}.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
