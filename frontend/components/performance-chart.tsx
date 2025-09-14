"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts"

interface PerformanceChartProps {
  data: Array<Record<string, any>>
  title: string
  type: "line" | "bar"
  dataKey: string
  xAxisKey: string
}

export function PerformanceChart({ data, title, type, dataKey, xAxisKey }: PerformanceChartProps) {
  const formatTooltipValue = (value: any, name: string) => {
    if (name === "accuracy") {
      return [`${value}%`, "Accuracy"]
    }
    if (name === "variance") {
      return [`${value}%`, "Variance Explained"]
    }
    return [value, name]
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-lg">{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            {type === "line" ? (
              <LineChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                <XAxis dataKey={xAxisKey} className="text-muted-foreground" fontSize={12} />
                <YAxis
                  className="text-muted-foreground"
                  fontSize={12}
                  domain={dataKey === "accuracy" ? [60, 85] : ["dataMin", "dataMax"]}
                />
                <Tooltip
                  formatter={formatTooltipValue}
                  labelFormatter={(label) => `Year: ${label}`}
                  contentStyle={{
                    backgroundColor: "hsl(var(--card))",
                    border: "1px solid hsl(var(--border))",
                    borderRadius: "8px",
                  }}
                />
                <Line
                  type="monotone"
                  dataKey={dataKey}
                  stroke="hsl(var(--primary))"
                  strokeWidth={3}
                  dot={{ fill: "hsl(var(--primary))", strokeWidth: 2, r: 4 }}
                  activeDot={{ r: 6, stroke: "hsl(var(--primary))", strokeWidth: 2 }}
                />
              </LineChart>
            ) : (
              <BarChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                <XAxis dataKey={xAxisKey} className="text-muted-foreground" fontSize={12} />
                <YAxis className="text-muted-foreground" fontSize={12} />
                <Tooltip
                  formatter={formatTooltipValue}
                  contentStyle={{
                    backgroundColor: "hsl(var(--card))",
                    border: "1px solid hsl(var(--border))",
                    borderRadius: "8px",
                  }}
                />
                <Bar dataKey={dataKey} fill="hsl(var(--primary))" radius={[4, 4, 0, 0]} />
              </BarChart>
            )}
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  )
}
