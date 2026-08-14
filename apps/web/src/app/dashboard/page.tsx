"use client"
import { useCountries } from "@/hooks/useCountries"
import dynamic from "next/dynamic"
import { ErrorState } from "@/components/ui/ErrorState"
import { EmptyState } from "@/components/ui/EmptyState"

// Map must be dynamically imported to avoid SSR window is not defined error
const WorldMap = dynamic(() => import('@/components/ui/WorldMap'), { 
  ssr: false, 
  loading: () => (
    <div className="w-full h-[600px] bg-gray-100 dark:bg-gray-800 animate-pulse rounded-lg border border-gray-200 dark:border-gray-800" />
  ) 
})

export default function DashboardPage() {
  const { data, isLoading, isError, error, refetch } = useCountries(0, 100)

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight text-gray-900 dark:text-white">Global Economic Map</h1>
        <p className="text-gray-500 dark:text-gray-400">
          Interactive view of global economies. Click a country to view its profile.
        </p>
      </div>
      
      {isLoading ? (
        <div className="w-full h-[600px] bg-gray-100 dark:bg-gray-800 animate-pulse rounded-lg border border-gray-200 dark:border-gray-800" />
      ) : isError ? (
        <ErrorState 
          title="Failed to load global map data" 
          error={error} 
          onRetry={refetch} 
        />
      ) : !data?.data || data.data.length === 0 ? (
        <EmptyState 
          title="No countries found" 
          description="There are currently no geographical economic profiles available in the database." 
        />
      ) : (
        <WorldMap countries={data.data} />
      )}
    </div>
  )
}

