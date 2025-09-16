"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { RefreshCw, ArrowLeft } from "lucide-react"
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line } from "recharts"
import type { PCAInfo, PCALoadings, PerformanceMetrics, FeatureImportanceResponse } from "@/lib/types"
import { mockPCAInfo, mockPCALoadings, mockPerformanceMetrics, mockFeatureImportance } from "@/lib/mock-data"
import Link from "next/link"

export default function AnalyticsPage() {
  const [pcaInfo, setPcaInfo] = useState<PCAInfo | null>(null)
  const [pcaLoadings, setPcaLoadings] = useState<PCALoadings | null>(null)
  const [performance, setPerformance] = useState<PerformanceMetrics | null>(null)
  const [featureImportance, setFeatureImportance] = useState<FeatureImportanceResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null)

  const loadAnalytics = async () => {
    setLoading(true)
    try {
      // Simulate API calls with mock data
      await new Promise((resolve) => setTimeout(resolve, 1000))

      setPcaInfo(mockPCAInfo)
      setPcaLoadings(mockPCALoadings)
      setPerformance(mockPerformanceMetrics)
      setFeatureImportance(mockFeatureImportance)
      setLastUpdated(new Date())
    } catch (error) {
      console.error("Failed to load analytics:", error)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadAnalytics()
  }, [])

  const varianceData =
    pcaInfo?.explainedVarianceRatio.map((ratio, index) => ({
      component: `PC${index + 1}`,
      variance: ratio * 100,
      cumulative: pcaInfo.cumulativeVarianceRatio[index] * 100,
    })) || []

  const importanceData =
    featureImportance?.items.map((item) => ({
      feature: item.feature.replace(/_/g, " "),
      importance: item.importance * 100,
    })) || []

  if (loading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="text-center space-y-4">
          <RefreshCw className="h-8 w-8 animate-spin mx-auto text-primary" />
          <p className="text-muted-foreground">Loading analytics...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b bg-card">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Link href="/">
                <Button variant="ghost" size="sm">
                  <ArrowLeft className="mr-2 h-4 w-4" />
                  Back
                </Button>
              </Link>
              <h1 className="text-3xl font-bold text-primary">Model Analytics</h1>
            </div>
            <div className="flex items-center gap-4">
              <div className="text-sm text-muted-foreground">Last updated: {lastUpdated?.toLocaleString()}</div>
              <Button onClick={loadAnalytics} variant="outline" size="sm">
                <RefreshCw className="mr-2 h-4 w-4" />
                Refresh
              </Button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8">
        <div className="space-y-8">
          {/* Performance Metrics */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">Accuracy</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-primary">
                  {((performance?.accuracy || 0) * 100).toFixed(1)}%
                </div>
                <Progress value={(performance?.accuracy || 0) * 100} className="mt-2" />
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">ROC AUC</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-primary">{(performance?.rocAuc || 0).toFixed(3)}</div>
                <Progress value={(performance?.rocAuc || 0) * 100} className="mt-2" />
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">Precision</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-primary">
                  {((performance?.precision || 0) * 100).toFixed(1)}%
                </div>
                <Progress value={(performance?.precision || 0) * 100} className="mt-2" />
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">F1 Score</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-primary">{(performance?.f1 || 0).toFixed(3)}</div>
                <Progress value={(performance?.f1 || 0) * 100} className="mt-2" />
              </CardContent>
            </Card>
          </div>

          {/* Confusion Matrix */}
          <Card>
            <CardHeader>
              <CardTitle>Confusion Matrix</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-4 max-w-md">
                <div className="text-center">
                  <div className="text-sm text-muted-foreground mb-2">True Negatives</div>
                  <Badge variant="outline" className="text-lg px-4 py-2">
                    {performance?.confusionMatrix[0][0]}
                  </Badge>
                </div>
                <div className="text-center">
                  <div className="text-sm text-muted-foreground mb-2">False Positives</div>
                  <Badge variant="destructive" className="text-lg px-4 py-2">
                    {performance?.confusionMatrix[0][1]}
                  </Badge>
                </div>
                <div className="text-center">
                  <div className="text-sm text-muted-foreground mb-2">False Negatives</div>
                  <Badge variant="destructive" className="text-lg px-4 py-2">
                    {performance?.confusionMatrix[1][0]}
                  </Badge>
                </div>
                <div className="text-center">
                  <div className="text-sm text-muted-foreground mb-2">True Positives</div>
                  <Badge variant="outline" className="text-lg px-4 py-2">
                    {performance?.confusionMatrix[1][1]}
                  </Badge>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* PCA Analysis */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>PCA Explained Variance</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={varianceData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="component" />
                      <YAxis />
                      <Tooltip formatter={(value: number) => [`${value.toFixed(1)}%`, "Variance"]} />
                      <Line type="monotone" dataKey="variance" stroke="hsl(var(--primary))" strokeWidth={2} />
                      <Line
                        type="monotone"
                        dataKey="cumulative"
                        stroke="hsl(var(--secondary))"
                        strokeWidth={2}
                        strokeDasharray="5 5"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                <div className="mt-4 text-sm text-muted-foreground">
                  Components: {pcaInfo?.nComponents} | Total Variance Explained:{" "}
                  {((pcaInfo?.cumulativeVarianceRatio[pcaInfo.nComponents - 1] || 0) * 100).toFixed(1)}%
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Feature Importance</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={importanceData} layout="horizontal">
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis type="number" />
                      <YAxis dataKey="feature" type="category" width={120} />
                      <Tooltip formatter={(value: number) => [`${value.toFixed(1)}%`, "Importance"]} />
                      <Bar dataKey="importance" fill="hsl(var(--primary))" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Feature Names */}
          <Card>
            <CardHeader>
              <CardTitle>Model Features</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2">
                {pcaLoadings?.featureNames.map((feature, index) => (
                  <Badge key={index} variant="outline" className="justify-center">
                    {feature.replace(/_/g, " ")}
                  </Badge>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  )
}
