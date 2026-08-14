"use client"
import React from "react"
import { AlertCircle, RefreshCw } from "lucide-react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

export interface ErrorStateProps {
  title?: string
  message?: string
  error?: unknown
  onRetry?: () => void
  className?: string
}

export function ErrorState({
  title = "Something went wrong",
  message,
  error,
  onRetry,
  className,
}: ErrorStateProps) {
  let displayMessage = message
  if (!displayMessage && error) {
    if (error instanceof Error) {
      displayMessage = error.message
    } else if (typeof error === "object" && error !== null && "message" in error) {
      displayMessage = String((error as { message: unknown }).message)
    } else if (typeof error === "string") {
      displayMessage = error
    } else {
      displayMessage = "An unexpected network or application error occurred while fetching data."
    }
  }

  return (
    <div
      className={cn(
        "flex flex-col items-center justify-center p-8 rounded-xl border border-red-200 dark:border-red-900/40 bg-red-50/50 dark:bg-red-950/10 text-center my-4 transition-all",
        className
      )}
    >
      <div className="w-12 h-12 rounded-full bg-red-100 dark:bg-red-900/30 flex items-center justify-center text-red-600 dark:text-red-400 mb-4 shadow-sm">
        <AlertCircle size={24} />
      </div>
      <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-1">
        {title}
      </h3>
      <p className="text-sm text-gray-600 dark:text-gray-400 max-w-md mb-6 leading-relaxed">
        {displayMessage || "We couldn't retrieve this information at the moment. Please check your network connection or try again."}
      </p>
      {onRetry && (
        <Button
          onClick={onRetry}
          variant="outline"
          className="flex items-center gap-2 border-red-200 dark:border-red-800 hover:bg-red-100 dark:hover:bg-red-900/30 text-red-700 dark:text-red-300 font-medium px-5 cursor-pointer"
        >
          <RefreshCw size={16} />
          <span>Try Again</span>
        </Button>
      )}
    </div>
  )
}
