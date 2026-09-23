"use client";

/**
 * EconoSphere AI — Phase 11 Step 13
 * ForecastComparisonTable — 5-country production ML forecast comparison.
 */

import React from "react";
import { GDPForecastResponse } from "@/hooks/useMLForecast";
import { TrendingUp, TrendingDown, FlaskConical } from "lucide-react";
import Link from "next/link";

interface ForecastComparisonTableProps {
  forecasts: GDPForecastResponse[];
}

function formatPct(val: number): string {
  const sign = val >= 0 ? "+" : "";
  return `${sign}${val.toFixed(2)}%`;
}

export function ForecastComparisonTable({ forecasts }: ForecastComparisonTableProps) {
  if (!forecasts || forecasts.length === 0) return null;

  return (
    <div className="bg-slate-900 border border-blue-900/40 rounded-2xl overflow-hidden shadow-xl shadow-blue-900/10">
      <div className="bg-gradient-to-r from-blue-950 via-slate-900 to-indigo-950 px-5 py-3 flex items-center justify-between border-b border-blue-900/30">
        <div className="flex items-center gap-2">
          <FlaskConical className="w-4 h-4 text-blue-400" />
          <span className="text-xs font-bold uppercase tracking-widest text-blue-300">
            Cross-Country Comparison (Production ML)
          </span>
        </div>
        <span className="px-2 py-0.5 text-[10px] font-extrabold uppercase tracking-wider rounded border border-amber-500/40 bg-amber-500/10 text-amber-300">
          FORECAST
        </span>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm text-slate-300">
          <thead className="text-xs uppercase bg-slate-950/50 text-slate-400 border-b border-slate-800">
            <tr>
              <th scope="col" className="px-6 py-4 font-semibold tracking-wider">Country</th>
              <th scope="col" className="px-6 py-4 font-semibold tracking-wider">Point Forecast</th>
              <th scope="col" className="px-6 py-4 font-semibold tracking-wider hidden sm:table-cell">90% Lower</th>
              <th scope="col" className="px-6 py-4 font-semibold tracking-wider hidden sm:table-cell">90% Upper</th>
              <th scope="col" className="px-6 py-4 font-semibold tracking-wider text-right">Interval Width</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800">
            {forecasts.map((f) => {
              const isPositive = f.predicted_gdp_growth >= 0;
              return (
                <tr key={f.country_code} className="hover:bg-slate-800/50 transition-colors group">
                  <td className="px-6 py-4 whitespace-nowrap">
                    <Link
                      href={`/dashboard/countries/${f.country_code}`}
                      className="flex items-center gap-2 group-hover:text-blue-400 transition-colors"
                    >
                      <span className="font-bold">{f.country_name}</span>
                      <span className="text-xs text-slate-500 font-mono">({f.country_code})</span>
                    </Link>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap font-mono font-bold">
                    <div className="flex items-center gap-2">
                      <span className={isPositive ? "text-emerald-400" : "text-red-400"}>
                        {formatPct(f.predicted_gdp_growth)}
                      </span>
                      {isPositive ? (
                        <TrendingUp className="w-4 h-4 text-emerald-400/50" />
                      ) : (
                        <TrendingDown className="w-4 h-4 text-red-400/50" />
                      )}
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap font-mono text-red-400/80 hidden sm:table-cell">
                    {formatPct(f.lower_bound_90)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap font-mono text-emerald-400/80 hidden sm:table-cell">
                    {formatPct(f.upper_bound_90)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap font-mono text-slate-400 text-right">
                    {formatPct(f.interval_width)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
