"use client"
import React, { useState } from "react"
import { useCountries, useSearchCountries } from "@/hooks/useCountries"
import Link from "next/link"
import { Globe, Users, DollarSign, Search, X, ChevronLeft, ChevronRight } from "lucide-react"
import { ErrorState } from "@/components/ui/ErrorState"
import { EmptyState } from "@/components/ui/EmptyState"
import { Button } from "@/components/ui/button"

export default function CountriesPage() {
  const [currentPage, setCurrentPage] = useState(1)
  const [pageSize] = useState(9)
  const [searchQuery, setSearchQuery] = useState("")

  const isSearching = searchQuery.trim().length >= 2

  const {
    data: pagedData,
    isLoading: pagedLoading,
    isError: pagedError,
    error: pagedErr,
    refetch: pagedRefetch,
  } = useCountries((currentPage - 1) * pageSize, pageSize)

  const {
    data: searchData,
    isLoading: searchLoading,
    isError: searchError,
    error: searchErr,
    refetch: searchRefetch,
  } = useSearchCountries(searchQuery)

  const isLoading = isSearching ? searchLoading : pagedLoading
  const isError = isSearching ? searchError : pagedError
  const error = isSearching ? searchErr : pagedErr
  const refetch = isSearching ? searchRefetch : pagedRefetch
  const countries = isSearching ? searchData || [] : pagedData?.data || []
  const totalCount = isSearching ? searchData?.length || 0 : pagedData?.total || 0
  const totalPages = isSearching ? 1 : Math.max(1, Math.ceil(totalCount / pageSize))

  return (
    <div className="space-y-6">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-gray-900 dark:text-white">Global Economies</h1>
          <p className="text-gray-500 dark:text-gray-400">
            Browse and analyze economic indicators across the world.
          </p>
        </div>

        {/* Local Filter / Search Bar */}
        <div className="relative w-full md:w-80">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" size={16} />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => {
              setSearchQuery(e.target.value)
              if (currentPage !== 1) setCurrentPage(1)
            }}
            placeholder="Search country name, code, or region..."
            className="w-full pl-9 pr-9 py-2 rounded-lg border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 shadow-xs transition-all"
          />
          {searchQuery && (
            <button
              onClick={() => setSearchQuery("")}
              className="absolute right-2.5 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 cursor-pointer p-1"
            >
              <X size={15} />
            </button>
          )}
        </div>
      </div>
      
      {isLoading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {[1, 2, 3, 4, 5, 6].map((i) => (
            <div key={i} className="h-52 bg-gray-100 dark:bg-gray-800 animate-pulse rounded-xl border border-gray-200 dark:border-gray-800" />
          ))}
        </div>
      ) : isError ? (
        <ErrorState
          title="Failed to load economies"
          error={error}
          onRetry={refetch}
        />
      ) : !countries || countries.length === 0 ? (
        <EmptyState
          title={isSearching ? `No economies match "${searchQuery}"` : "No global economies found"}
          description={
            isSearching
              ? "Try adjusting your search keywords or clearing the filter."
              : "The dataset currently does not contain any country intelligence profiles."
          }
          action={
            isSearching
              ? { label: "Clear Search", onClick: () => setSearchQuery("") }
              : undefined
          }
        />
      ) : (
        <div className="space-y-8">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {countries.map((country) => (
              <Link 
                href={`/dashboard/countries/${country.iso3}`} 
                key={country.iso3}
                className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-xl p-6 hover:shadow-md transition-shadow cursor-pointer block group"
              >
                <div className="flex justify-between items-start mb-4">
                  <div>
                    <h3 className="text-xl font-bold text-gray-900 dark:text-gray-100 group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors">
                      {country.name}
                    </h3>
                    <span className="text-sm text-gray-500 font-mono">{country.iso3}</span>
                  </div>
                  {country.metadata_info?.flag_url ? (
                    /* eslint-disable-next-line @next/next/no-img-element */
                    <img
                      src={country.metadata_info.flag_url}
                      alt={`${country.name} flag`}
                      className="w-10 h-7 object-cover rounded shadow-xs"
                    />
                  ) : (
                    <div className="w-10 h-7 bg-gray-100 dark:bg-gray-800 rounded flex items-center justify-center">
                      <Globe size={16} className="text-gray-400" />
                    </div>
                  )}
                </div>
                
                <div className="space-y-3 pt-2 border-t border-gray-100 dark:border-gray-800/60">
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-gray-500 flex items-center gap-1.5"><Globe size={15}/> Region</span>
                    <span className="font-medium text-gray-700 dark:text-gray-300">{country.region || "Unknown"}</span>
                  </div>
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-gray-500 flex items-center gap-1.5"><Users size={15}/> Population</span>
                    <span className="font-medium text-gray-700 dark:text-gray-300">
                      {country.metadata_info?.population
                        ? country.metadata_info.population.toLocaleString()
                        : "N/A"}
                    </span>
                  </div>
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-gray-500 flex items-center gap-1.5"><DollarSign size={15}/> Currency</span>
                    <span className="font-medium font-mono text-gray-700 dark:text-gray-300">
                      {country.metadata_info?.currency || "N/A"}
                    </span>
                  </div>
                </div>
              </Link>
            ))}
          </div>

          {/* Pagination Controls */}
          {!isSearching && totalPages > 1 && (
            <div className="flex items-center justify-between pt-6 border-t border-gray-200 dark:border-gray-800 text-sm text-gray-500 dark:text-gray-400">
              <div>
                Showing <span className="font-medium text-gray-900 dark:text-white">{(currentPage - 1) * pageSize + 1}</span> to{" "}
                <span className="font-medium text-gray-900 dark:text-white">{Math.min(currentPage * pageSize, totalCount)}</span> of{" "}
                <span className="font-medium text-gray-900 dark:text-white">{totalCount}</span> economies
              </div>
              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                  disabled={currentPage === 1}
                  className="flex items-center gap-1 cursor-pointer"
                >
                  <ChevronLeft size={16} />
                  <span>Previous</span>
                </Button>
                <span className="px-3 py-1 font-mono font-medium rounded bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300">
                  {currentPage} / {totalPages}
                </span>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
                  disabled={currentPage >= totalPages}
                  className="flex items-center gap-1 cursor-pointer"
                >
                  <span>Next</span>
                  <ChevronRight size={16} />
                </Button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

