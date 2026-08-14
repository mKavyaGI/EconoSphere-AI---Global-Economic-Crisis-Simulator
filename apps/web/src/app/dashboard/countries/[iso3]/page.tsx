"use client"
import { useParams } from "next/navigation"
import { useCountry, useCountryHistory } from "@/hooks/useCountry"
import { useRisk } from "@/hooks/useAI"
import { LineChart } from "@/components/charts/LineChart"
import { ArrowLeft, Users, Landmark, MapPin, DollarSign, Target, Activity, Brain, Globe } from "lucide-react"
import Link from "next/link"
import { ErrorState } from "@/components/ui/ErrorState"
import { EmptyState } from "@/components/ui/EmptyState"

export default function CountryProfilePage() {
  const { iso3 } = useParams()
  const {
    data: country,
    isLoading: isCountryLoading,
    isError: isCountryError,
    error: countryError,
    refetch: refetchCountry,
  } = useCountry(iso3 as string)

  const {
    data: gdpHistory,
    isLoading: isGdpLoading,
    isError: isGdpError,
    error: gdpError,
    refetch: refetchGdp,
  } = useCountryHistory(iso3 as string, "GDP")

  const {
    data: infHistory,
    isLoading: isInfLoading,
    isError: isInfError,
    error: infError,
    refetch: refetchInf,
  } = useCountryHistory(iso3 as string, "Inflation")

  const { data: riskScorecard } = useRisk(iso3 as string)

  if (isCountryLoading) {
    return (
      <div className="animate-pulse space-y-6">
        <div className="h-20 bg-gray-100 dark:bg-gray-800 rounded-lg" />
        <div className="h-64 bg-gray-100 dark:bg-gray-800 rounded-lg" />
      </div>
    )
  }

  if (isCountryError) {
    return (
      <div className="space-y-6">
        <Link href="/dashboard/countries" className="text-blue-600 dark:text-blue-400 hover:underline flex items-center gap-2 mb-4 text-sm font-medium">
          <ArrowLeft size={16} /> Back to Countries
        </Link>
        <ErrorState title="Failed to load country profile" error={countryError} onRetry={refetchCountry} />
      </div>
    )
  }

  if (!country) {
    return (
      <div className="space-y-6">
        <Link href="/dashboard/countries" className="text-blue-600 dark:text-blue-400 hover:underline flex items-center gap-2 mb-4 text-sm font-medium">
          <ArrowLeft size={16} /> Back to Countries
        </Link>
        <EmptyState
          title="Country not found"
          description={`No country intelligence record matches the ISO code "${iso3}".`}
        />
      </div>
    )
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <Link href="/dashboard/countries" className="text-blue-600 dark:text-blue-400 hover:underline flex items-center gap-2 mb-4 text-sm font-medium">
          <ArrowLeft size={16} /> Back to Countries
        </Link>
        <div className="flex items-center gap-4">
          {country.metadata_info?.flag_url ? (
            /* eslint-disable-next-line @next/next/no-img-element */
            <img
              src={country.metadata_info.flag_url}
              alt={`${country.name} flag`}
              className="w-16 h-12 object-cover rounded shadow-md border border-gray-200 dark:border-gray-800"
            />
          ) : (
            <div className="w-16 h-12 bg-gray-100 dark:bg-gray-800 rounded flex items-center justify-center">
              <Globe size={24} className="text-gray-400" />
            </div>
          )}
          <div>
            <h1 className="text-4xl font-bold tracking-tight text-gray-900 dark:text-white">{country.name}</h1>
            <p className="text-gray-500 dark:text-gray-400 flex items-center gap-2 mt-1">
              <MapPin size={16}/> {country.region} • {country.income_group}
            </p>
          </div>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-xl p-5 shadow-sm">
          <div className="flex justify-between items-center mb-2 text-gray-500">
            <span className="text-sm font-medium">Population</span>
            <Users size={18}/>
          </div>
          <div className="text-2xl font-bold text-gray-900 dark:text-gray-100">
            {country.metadata_info?.population ? country.metadata_info.population.toLocaleString() : "N/A"}
          </div>
        </div>
        <div className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-xl p-5 shadow-sm">
          <div className="flex justify-between items-center mb-2 text-gray-500">
            <span className="text-sm font-medium">Capital</span>
            <Landmark size={18}/>
          </div>
          <div className="text-2xl font-bold text-gray-900 dark:text-gray-100">
            {country.metadata_info?.capital || "N/A"}
          </div>
        </div>
        <div className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-xl p-5 shadow-sm">
          <div className="flex justify-between items-center mb-2 text-gray-500">
            <span className="text-sm font-medium">Currency</span>
            <DollarSign size={18}/>
          </div>
          <div className="text-2xl font-bold font-mono text-gray-900 dark:text-gray-100">
            {country.metadata_info?.currency || "N/A"}
          </div>
        </div>
        <div className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-xl p-5 shadow-sm">
          <div className="flex justify-between items-center mb-2 text-gray-500">
            <span className="text-sm font-medium">Risk Score</span>
            <Target size={18}/>
          </div>
          <div className="text-2xl font-bold text-emerald-600 dark:text-emerald-400">
            {riskScorecard?.overall_risk_tier || "Low / Stable"}
          </div>
        </div>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-xl p-6 shadow-sm flex flex-col justify-center min-h-[380px]">
          {isGdpLoading ? (
            <div className="h-[350px] animate-pulse bg-gray-100 dark:bg-gray-800 rounded-lg" />
          ) : isGdpError ? (
            <ErrorState title="Failed to load GDP trajectory" error={gdpError} onRetry={refetchGdp} />
          ) : !gdpHistory || gdpHistory.length === 0 ? (
            <EmptyState title="No GDP Data Available" description="Historical GDP time series is currently not indexed for this economy." />
          ) : (
            <LineChart title="GDP Growth (Trillions USD)" data={gdpHistory} xKey="year" yKey="value" />
          )}
        </div>
        <div className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-xl p-6 shadow-sm flex flex-col justify-center min-h-[380px]">
          {isInfLoading ? (
            <div className="h-[350px] animate-pulse bg-gray-100 dark:bg-gray-800 rounded-lg" />
          ) : isInfError ? (
            <ErrorState title="Failed to load Inflation trajectories" error={infError} onRetry={refetchInf} />
          ) : !infHistory || infHistory.length === 0 ? (
            <EmptyState title="No Inflation Data Available" description="Historical Inflation time series is currently not indexed for this economy." />
          ) : (
            <LineChart title="Inflation Rate (%)" data={infHistory} xKey="year" yKey="value" />
          )}
        </div>
      </div>
      
      {/* Active Analytical Modules */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Link href="/dashboard/trade/network" className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 hover:border-blue-500 dark:hover:border-blue-500 rounded-xl p-6 flex flex-col items-center justify-center text-center h-48 group transition-all shadow-sm hover:shadow-md">
           <Globe className="text-blue-500 mb-3 group-hover:scale-110 transition-transform" size={32}/>
           <h3 className="font-bold text-gray-900 dark:text-white text-lg">Trade Network Linkages</h3>
           <p className="text-xs text-gray-500 dark:text-gray-400 mt-1 max-w-[200px]">Explore bilateral tariffs and trade volumes in the live interactive graph.</p>
        </Link>
        <Link href="/dashboard/simulation" className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 hover:border-emerald-500 dark:hover:border-emerald-500 rounded-xl p-6 flex flex-col items-center justify-center text-center h-48 group transition-all shadow-sm hover:shadow-md">
           <Activity className="text-emerald-500 mb-3 group-hover:scale-110 transition-transform" size={32}/>
           <h3 className="font-bold text-gray-900 dark:text-white text-lg">Macro Stress Test</h3>
           <p className="text-xs text-gray-500 dark:text-gray-400 mt-1 max-w-[200px]">Run tariff shocks and financial crisis propagation simulations.</p>
        </Link>
        <Link href="/dashboard/forecast" className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 hover:border-purple-500 dark:hover:border-purple-500 rounded-xl p-6 flex flex-col items-center justify-center text-center h-48 group transition-all shadow-sm hover:shadow-md">
           <Brain className="text-purple-500 mb-3 group-hover:scale-110 transition-transform" size={32}/>
           <h3 className="font-bold text-gray-900 dark:text-white text-lg">AI Policy &amp; Forecast</h3>
           <p className="text-xs text-gray-500 dark:text-gray-400 mt-1 max-w-[200px]">View explainable economic projections and policy recommendations.</p>
        </Link>
      </div>

    </div>
  )
}

