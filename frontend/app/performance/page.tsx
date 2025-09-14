"use client"

import { Navigation } from "@/components/navigation"
import { MetricCard } from "@/components/metric-card"
import { PerformanceChart } from "@/components/performance-chart"
import { FeatureImportanceChart } from "@/components/feature-importance-chart"
import { ConfusionMatrix } from "@/components/confusion-matrix"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Target, TrendingUp, Zap, Database } from "lucide-react"

// Mock performance data
const performanceMetrics = {
  test_accuracy: 78.8,
  roc_auc: 88.4,
  generalization_gap: 6.9,
  feature_count: 52,
}

const historicalAccuracy = [
  { year: 2015, accuracy: 65.2 },
  { year: 2016, accuracy: 68.1 },
  { year: 2017, accuracy: 71.3 },
  { year: 2018, accuracy: 73.8 },
  { year: 2019, accuracy: 75.2 },
  { year: 2020, accuracy: 76.9 },
  { year: 2021, accuracy: 77.1 },
  { year: 2022, accuracy: 77.8 },
  { year: 2023, accuracy: 78.2 },
  { year: 2024, accuracy: 78.8 },
]

const featureImportance = [
  { feature: "EPA per Play Differential", importance: 0.18 },
  { feature: "Recent Form Score", importance: 0.15 },
  { feature: "Strength of Schedule", importance: 0.12 },
  { feature: "Turnover Differential", importance: 0.11 },
  { feature: "Red Zone Efficiency", importance: 0.09 },
  { feature: "Third Down Conversion", importance: 0.08 },
  { feature: "Time of Possession", importance: 0.07 },
  { feature: "Penalty Yards", importance: 0.06 },
  { feature: "Home Field Advantage", importance: 0.05 },
  { feature: "Weather Conditions", importance: 0.04 },
]

const pcaComponents = [
  { component: "PC1", variance: 23.4 },
  { component: "PC2", variance: 18.7 },
  { component: "PC3", variance: 14.2 },
  { component: "PC4", variance: 11.8 },
  { component: "PC5", variance: 9.3 },
  { component: "PC6", variance: 7.6 },
  { component: "PC7", variance: 6.1 },
  { component: "PC8", variance: 4.9 },
]

export default function ModelPerformancePage() {
  return (
    <div className="min-h-screen bg-background">
      <Navigation />

      <main className="container mx-auto px-4 py-8">
        <div className="space-y-8">
          {/* Header */}
          <div className="text-center space-y-2">
            <h1 className="text-3xl font-bold text-foreground">Model Performance Dashboard</h1>
            <p className="text-muted-foreground">2015-2024 Testing Period</p>
          </div>

          {/* Key Metrics */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <MetricCard
              title="Test Accuracy"
              value={`${performanceMetrics.test_accuracy}%`}
              subtitle="Out-of-sample performance"
              trend="up"
              icon={<Target className="h-5 w-5" />}
            />
            <MetricCard
              title="ROC-AUC"
              value={`${performanceMetrics.roc_auc}%`}
              subtitle="Area under curve"
              trend="up"
              icon={<TrendingUp className="h-5 w-5" />}
            />
            <MetricCard
              title="Generalization Gap"
              value={`${performanceMetrics.generalization_gap}%`}
              subtitle="Train vs test difference"
              trend="down"
              icon={<Zap className="h-5 w-5" />}
            />
            <MetricCard
              title="Features Used"
              value={performanceMetrics.feature_count}
              subtitle="After feature selection"
              trend="neutral"
              icon={<Database className="h-5 w-5" />}
            />
          </div>

          {/* Charts Section */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Historical Accuracy */}
            <PerformanceChart
              data={historicalAccuracy}
              title="Model Accuracy Over Time"
              type="line"
              dataKey="accuracy"
              xAxisKey="year"
            />

            {/* PCA Components */}
            <PerformanceChart
              data={pcaComponents}
              title="PCA Components (96% Variance Explained)"
              type="bar"
              dataKey="variance"
              xAxisKey="component"
            />

            {/* Feature Importance */}
            <FeatureImportanceChart data={featureImportance} title="Top 10 Feature Importance" />

            {/* Confusion Matrix */}
            <ConfusionMatrix />
          </div>

          {/* Technical Details */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Phase Comparison */}
            <Card>
              <CardHeader>
                <CardTitle>Model Evolution</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex items-center justify-between p-4 bg-muted/50 rounded-lg">
                    <div>
                      <h4 className="font-semibold">Phase 1.2</h4>
                      <p className="text-sm text-muted-foreground">Initial baseline model</p>
                    </div>
                    <Badge variant="secondary">72.3% Accuracy</Badge>
                  </div>
                  <div className="flex items-center justify-between p-4 bg-primary/5 rounded-lg border-2 border-primary/20">
                    <div>
                      <h4 className="font-semibold">Phase 3.0</h4>
                      <p className="text-sm text-muted-foreground">Current production model</p>
                    </div>
                    <Badge className="bg-primary text-primary-foreground">78.8% Accuracy</Badge>
                  </div>
                  <div className="text-center pt-2">
                    <Badge variant="outline" className="text-green-600 border-green-600">
                      +6.5% Improvement
                    </Badge>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Model Architecture */}
            <Card>
              <CardHeader>
                <CardTitle>Model Architecture</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <p className="text-sm font-medium">Algorithm</p>
                      <p className="text-sm text-muted-foreground">Gradient Boosting</p>
                    </div>
                    <div className="space-y-2">
                      <p className="text-sm font-medium">Validation</p>
                      <p className="text-sm text-muted-foreground">5-Fold CV</p>
                    </div>
                    <div className="space-y-2">
                      <p className="text-sm font-medium">Hyperparameters</p>
                      <p className="text-sm text-muted-foreground">Bayesian Optimized</p>
                    </div>
                    <div className="space-y-2">
                      <p className="text-sm font-medium">Training Data</p>
                      <p className="text-sm text-muted-foreground">2015-2023 Seasons</p>
                    </div>
                  </div>
                  <div className="pt-4 border-t">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium">Cross-Validation Score</span>
                      <Badge variant="outline">77.2% ± 2.1%</Badge>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </main>
    </div>
  )
}
