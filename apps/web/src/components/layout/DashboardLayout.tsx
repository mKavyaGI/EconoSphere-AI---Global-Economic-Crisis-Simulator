"use client"
import { Sidebar } from "./Sidebar"
import { Topbar } from "./Topbar"

export function DashboardLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex h-screen overflow-hidden bg-gray-50 dark:bg-gray-950 text-gray-900 dark:text-gray-100">
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        <Topbar />
        <main className="flex-1 overflow-x-hidden overflow-y-auto">
          <div className="container mx-auto p-6 md:p-8 max-w-7xl">
            {children}
          </div>
        </main>
      </div>
    </div>
  )
}
