"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts"

interface FeatureImportanceChartProps {
  data: Array<{ feature: string; importance: number }>
  title: string
}

export function FeatureImportanceChart({ data, title }: FeatureImportanceChartProps) {
  // Sort data by importance and take top 10
  const sortedData = data
    .sort((a, b) => b.importance - a.importance)
    .slice(0, 10)
    .map((item) => ({
      ...item,
      shortFeature: item.feature.length > 20 ? `${item.feature.substring(0, 17)}...` : item.feature,
    }))

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-lg">{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={sortedData} layout="horizontal" margin={{ top: 5, right: 30, left: 100, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
              <XAxis
                type="number"
                className="text-muted-foreground"
                fontSize={12}
                tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
              />
              <YAxis
                type="category"
                dataKey="shortFeature"
                className="text-muted-foreground"
                fontSize={11}
                width={90}
              />
              <Tooltip
                formatter={(value: number) => [`${(value * 100).toFixed(1)}%`, "Importance"]}
                labelFormatter={(label) => {
                  const fullFeature =
                    data.find((item) => item.feature.startsWith(label.replace("...", "")))?.feature || label
                  return fullFeature
                }}
                contentStyle={{
                  backgroundColor: "hsl(var(--card))",
                  border: "1px solid hsl(var(--border))",
                  borderRadius: "8px",
                }}
              />
              <Bar dataKey="importance" fill="hsl(var(--primary))" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  )
}
