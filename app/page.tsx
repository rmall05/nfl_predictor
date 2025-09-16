"use client"

import { useState } from "react"
import { PredictionForm } from "@/components/prediction-form"
import { PredictionResult } from "@/components/prediction-result"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { BarChart3, History, TrendingUp } from "lucide-react"
import type { PredictRequest, PredictResponse } from "@/lib/types"
import { mockPrediction } from "@/lib/mock-data"
import { HistoryManager } from "@/lib/history"
import { useToast } from "@/hooks/use-toast"
import Link from "next/link"

export default function HomePage() {
  const [prediction, setPrediction] = useState<PredictResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const { toast } = useToast()

  const handlePrediction = async (request: PredictRequest) => {
    setLoading(true)
    try {
      // Simulate API call with mock data
      await new Promise((resolve) => setTimeout(resolve, 1500))

      const result: PredictResponse = {
        ...mockPrediction,
        gameId: `${request.season}_${request.week}_${request.homeTeam}_${request.awayTeam}`,
        homeTeam: request.homeTeam,
        awayTeam: request.awayTeam,
        timestamp: new Date().toISOString(),
      }

      setPrediction(result)
      HistoryManager.addPrediction(result)

      toast({
        title: "Prediction generated",
        description: "Your NFL game prediction is ready!",
      })
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to generate prediction. Please try again.",
        variant: "destructive",
      })
    } finally {
      setLoading(false)
    }
  }

  const resetPrediction = () => {
    setPrediction(null)
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b bg-card">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <h1 className="text-3xl font-bold text-primary">NFL Predictor</h1>
            <nav className="flex items-center gap-4">
              <Link href="/analytics">
                <Button variant="ghost" size="sm">
                  <BarChart3 className="mr-2 h-4 w-4" />
                  Analytics
                </Button>
              </Link>
              <Link href="/history">
                <Button variant="ghost" size="sm">
                  <History className="mr-2 h-4 w-4" />
                  History
                </Button>
              </Link>
            </nav>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8">
        <div className="max-w-4xl mx-auto space-y-8">
          {!prediction ? (
            <>
              {/* Hero Section */}
              <div className="text-center space-y-4">
                <h2 className="text-4xl font-bold text-balance">Predict NFL Game Outcomes</h2>
                <p className="text-xl text-muted-foreground text-pretty max-w-2xl mx-auto">
                  Use advanced machine learning models to predict NFL game winners, margins, and key performance
                  factors.
                </p>
              </div>

              {/* Features */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <Card>
                  <CardHeader>
                    <TrendingUp className="h-8 w-8 text-primary mb-2" />
                    <CardTitle>Win Probabilities</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-muted-foreground">
                      Get precise win probability percentages for both teams based on historical data and current form.
                    </p>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <BarChart3 className="h-8 w-8 text-primary mb-2" />
                    <CardTitle>Predicted Margins</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-muted-foreground">
                      See the expected point differential and understand which team has the statistical advantage.
                    </p>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <History className="h-8 w-8 text-primary mb-2" />
                    <CardTitle>Key Factors</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-muted-foreground">
                      Discover the most influential statistics and metrics driving each prediction.
                    </p>
                  </CardContent>
                </Card>
              </div>

              {/* Prediction Form */}
              <div className="flex justify-center">
                <PredictionForm onSubmit={handlePrediction} loading={loading} />
              </div>
            </>
          ) : (
            <>
              {/* Results */}
              <div className="flex justify-center">
                <PredictionResult prediction={prediction} />
              </div>

              {/* New Prediction Button */}
              <div className="text-center">
                <Button onClick={resetPrediction} size="lg">
                  Make Another Prediction
                </Button>
              </div>
            </>
          )}
        </div>
      </main>
    </div>
  )
}
