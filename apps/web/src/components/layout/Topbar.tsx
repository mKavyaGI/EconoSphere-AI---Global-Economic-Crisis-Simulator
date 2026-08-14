"use client"
import React, { useState, useRef, useEffect } from "react"
import { Search, Bell, Globe, X, Loader2 } from "lucide-react"
import { ThemeToggle } from "@/components/ui/ThemeToggle"
import { useSearchCountries } from "@/hooks/useCountries"
import { useRouter } from "next/navigation"

export function Topbar() {
  const [query, setQuery] = useState("")
  const [isOpen, setIsOpen] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)
  const router = useRouter()

  const { data: searchResults, isLoading, isError } = useSearchCountries(query)

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setIsOpen(false)
      }
    }
    document.addEventListener("mousedown", handleClickOutside)
    return () => document.removeEventListener("mousedown", handleClickOutside)
  }, [])

  const handleSelectCountry = (iso3: string) => {
    router.push(`/dashboard/countries/${iso3}`)
    setIsOpen(false)
    setQuery("")
  }

  return (
    <header className="h-16 border-b border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-950 flex items-center justify-between px-6 z-20 relative shadow-sm">
      <div className="flex-1 flex items-center">
        <div ref={containerRef} className="relative w-96 max-w-md hidden md:block">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" size={16} />
          <input
            type="text"
            value={query}
            onChange={(e) => {
              setQuery(e.target.value)
              setIsOpen(e.target.value.trim().length >= 2)
            }}
            onFocus={() => {
              if (query.trim().length >= 2) setIsOpen(true)
            }}
            onKeyDown={(e) => {
              if (e.key === "Escape") setIsOpen(false)
              if (e.key === "Enter" && searchResults && searchResults.length > 0 && isOpen) {
                handleSelectCountry(searchResults[0].iso3)
              }
            }}
            placeholder="Search countries, simulations..."
            className="w-full pl-9 pr-8 py-1.5 rounded-md border border-gray-300 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
          />
          {query && (
            <button
              onClick={() => {
                setQuery("")
                setIsOpen(false)
              }}
              className="absolute right-2.5 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-200"
            >
              <X size={14} />
            </button>
          )}

          {/* Search Dropdown Results */}
          {isOpen && (
            <div className="absolute left-0 right-0 top-full mt-2 bg-white dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-800 shadow-xl overflow-hidden z-50 max-h-96 overflow-y-auto">
              {isLoading ? (
                <div className="p-4 text-center text-xs text-gray-500 flex items-center justify-center gap-2">
                  <Loader2 size={16} className="animate-spin text-blue-500" />
                  <span>Searching economies...</span>
                </div>
              ) : isError ? (
                <div className="p-4 text-center text-xs text-red-500">
                  Failed to fetch matching results.
                </div>
              ) : searchResults && searchResults.length > 0 ? (
                <div className="py-1">
                  <div className="px-3 py-1.5 text-[11px] font-semibold tracking-wider uppercase text-gray-400 bg-gray-50/50 dark:bg-gray-800/50 border-b border-gray-100 dark:border-gray-800/80">
                    Countries &amp; Territories ({searchResults.length})
                  </div>
                  {searchResults.map((country) => (
                    <div
                      key={country.iso3}
                      onClick={() => handleSelectCountry(country.iso3)}
                      className="flex items-center justify-between px-3 py-2.5 hover:bg-blue-50 dark:hover:bg-blue-900/20 cursor-pointer transition-colors border-b last:border-0 border-gray-100 dark:border-gray-800/50"
                    >
                      <div className="flex items-center gap-3">
                        {country.metadata_info?.flag_url ? (
                          /* eslint-disable-next-line @next/next/no-img-element */
                          <img
                            src={country.metadata_info.flag_url}
                            alt={country.name}
                            className="w-7 h-5 object-cover rounded shadow-xs"
                          />
                        ) : (
                          <Globe size={18} className="text-gray-400" />
                        )}
                        <div>
                          <div className="text-sm font-medium text-gray-900 dark:text-gray-100">
                            {country.name}
                          </div>
                          <div className="text-xs text-gray-400">{country.region || "Global"}</div>
                        </div>
                      </div>
                      <span className="text-xs font-mono px-2 py-0.5 rounded bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-300">
                        {country.iso3}
                      </span>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="p-6 text-center">
                  <p className="text-xs font-medium text-gray-500 dark:text-gray-400">
                    No matching countries found for &ldquo;{query}&rdquo;
                  </p>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
      <div className="flex items-center gap-4 text-gray-500 dark:text-gray-400">
        <ThemeToggle />
        <button className="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-full transition-colors relative cursor-pointer" aria-label="Notifications">
          <Bell size={18} />
          <span className="absolute top-1.5 right-1.5 w-2 h-2 bg-red-500 rounded-full"></span>
        </button>
        <div className="w-8 h-8 rounded-full bg-blue-100 dark:bg-blue-900/30 flex items-center justify-center border border-blue-200 dark:border-blue-800 cursor-pointer text-blue-700 dark:text-blue-400 font-medium text-sm">
          NM
        </div>
      </div>
    </header>
  )
}
