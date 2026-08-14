'use client'
import Link from 'next/link'
import { Globe, LineChart, Brain, Cpu, FileText, Network } from 'lucide-react'
import { usePathname } from 'next/navigation'

export function Sidebar() {
  const pathname = usePathname()
  
  const navItems = [
    { name: 'Map Overview', href: '/dashboard', icon: Globe },
    { name: 'Countries', href: '/dashboard/countries', icon: LineChart },
    { name: 'Trade Network', href: '/dashboard/trade', icon: Network },
    { name: 'Scenarios', href: '/dashboard/scenarios', icon: FileText },
    { name: 'Simulations', href: '/dashboard/simulation', icon: Cpu },
    { name: 'AI Forecast', href: '/dashboard/forecast', icon: Brain },
  ]


  return (
    <div className="flex flex-col w-64 bg-white dark:bg-gray-900 text-gray-900 dark:text-white min-h-screen border-r border-gray-200 dark:border-gray-800 shadow-sm z-10">
      <div className="flex items-center justify-center h-16 border-b border-gray-200 dark:border-gray-800">
        <span className="text-xl font-bold tracking-tight text-blue-600 dark:text-blue-400">EconoSphere AI</span>
      </div>
      <nav className="flex-1 px-4 py-6 space-y-1">
        {navItems.map((item) => {
          const Icon = item.icon
          const isActive = pathname === item.href || pathname.startsWith(item.href + '/')
          return (
            <Link 
              key={item.name} 
              href={item.href}
              className={`flex items-center gap-3 px-3 py-2.5 rounded-md transition-colors ${
                isActive 
                ? 'bg-blue-50 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400 font-medium' 
                : 'hover:bg-gray-100 dark:hover:bg-gray-800 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200'
              }`}
            >
              <Icon size={18} className={isActive ? 'text-blue-600 dark:text-blue-400' : 'text-gray-400 dark:text-gray-500'} />
              <span>{item.name}</span>
            </Link>
          )
        })}
      </nav>
    </div>
  )
}
