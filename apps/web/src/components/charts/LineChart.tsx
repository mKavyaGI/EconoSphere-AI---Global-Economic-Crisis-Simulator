"use client"
import ReactECharts from 'echarts-for-react'
import { useTheme } from 'next-themes'
import { useEffect, useState } from 'react'

interface LineChartProps {
  title: string
  data: unknown[]
  xKey: string
  yKey: string
}

export function LineChart({ title, data, xKey, yKey }: LineChartProps) {
  const { theme, systemTheme } = useTheme()
  const [mounted, setMounted] = useState(false)
  
  useEffect(() => {
    const id = requestAnimationFrame(() => setMounted(true))
    return () => cancelAnimationFrame(id)
  }, [])
  
  const currentTheme = theme === 'system' ? systemTheme : theme
  const isDark = currentTheme === 'dark'

  if (!mounted) return <div className="h-[350px] w-full animate-pulse bg-gray-100 dark:bg-gray-800 rounded-md" />

  const option = {
    backgroundColor: 'transparent',
    title: {
      text: title,
      textStyle: { color: isDark ? '#fff' : '#000', fontSize: 16 }
    },
    tooltip: { trigger: 'axis' },
    xAxis: {
      type: 'category',
      data: data.map(d => (d as Record<string, unknown>)[xKey]),
      axisLabel: { color: isDark ? '#aaa' : '#333' }
    },
    yAxis: {
      type: 'value',
      axisLabel: { color: isDark ? '#aaa' : '#333' },
      splitLine: { lineStyle: { color: isDark ? '#333' : '#eee' } }
    },
    series: [
      {
        data: data.map(d => (d as Record<string, unknown>)[yKey]),
        type: 'line',
        smooth: true,
        lineStyle: { width: 3, color: '#3b82f6' },
        itemStyle: { color: '#3b82f6' },
        areaStyle: {
          color: {
            type: 'linear',
            x: 0, y: 0, x2: 0, y2: 1,
            colorStops: [
              { offset: 0, color: 'rgba(59, 130, 246, 0.5)' },
              { offset: 1, color: 'rgba(59, 130, 246, 0.0)' }
            ]
          }
        }
      }
    ]
  }

  return <ReactECharts option={option} style={{ height: '350px', width: '100%' }} />
}
