"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { ArrowLeft, Trash2, RefreshCw } from "lucide-react"
import type { PredictionHistory } from "@/lib/types"
import { HistoryManager } from "@/lib/history"
import { mockTeams } from "@/lib/mock-data"
import Link from "next/link"

export default function HistoryPage() {
  const [history, setHistory] = useState<PredictionHistory[]>([])
  const [loading, setLoading] = useState(true)

  const loadHistory = () => {
    setLoading(true)
    const historyData = HistoryManager.getHistory()
    setHistory(historyData)
    setLoading(false)
  }

  useEffect(() => {
    loadHistory()
  }, [])

  const clearHistory = () => {
    HistoryManager.clearHistory()
    setHistory([])
  }

  const removePrediction = (id: string) => {
    HistoryManager.removePrediction(id)
    loadHistory()
  }

  const getTeamName = (abbr: string) => {
    return mockTeams.find((team) => team.abbr === abbr)?.name || abbr
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="text-center space-y-4">
          <RefreshCw className="h-8 w-8 animate-spin mx-auto text-primary" />
          <p className="text-muted-foreground">Loading history...</p>
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
              <h1 className="text-3xl font-bold text-primary">Prediction History</h1>
            </div>
            <div className="flex items-center gap-2">
              <Button onClick={loadHistory} variant="outline" size="sm">
                <RefreshCw className="mr-2 h-4 w-4" />
                Refresh
              </Button>
              {history.length > 0 && (
                <Button onClick={clearHistory} variant="destructive" size="sm">
                  <Trash2 className="mr-2 h-4 w-4" />
                  Clear All
                </Button>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8">
        {history.length === 0 ? (
          <div className="text-center py-12">
            <div className="text-6xl mb-4">📊</div>
            <h2 className="text-2xl font-bold mb-2">No Predictions Yet</h2>
            <p className="text-muted-foreground mb-6">
              Your prediction history will appear here once you start making predictions.
            </p>
            <Link href="/">
              <Button>Make Your First Prediction</Button>
            </Link>
          </div>
        ) : (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-semibold">
                {history.length} Prediction{history.length !== 1 ? "s" : ""}
              </h2>
            </div>

            <div className="grid gap-4">
              {history.map((item) => (
                <Card key={item.id}>
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-lg">
                        {getTeamName(item.prediction.awayTeam)} @ {getTeamName(item.prediction.homeTeam)}
                      </CardTitle>
                      <Button variant="ghost" size="sm" onClick={() => removePrediction(item.id)}>
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <div>
                        <div className="text-sm text-muted-foreground">Predicted Winner</div>
                        <div className="font-semibold text-primary">
                          {item.prediction.predictedWinner === "home"
                            ? getTeamName(item.prediction.homeTeam)
                            : getTeamName(item.prediction.awayTeam)}
                        </div>
                      </div>

                      <div>
                        <div className="text-sm text-muted-foreground">Margin</div>
                        <div className="font-semibold">
                          {Math.abs(item.prediction.predictedMargin).toFixed(1)} points
                        </div>
                      </div>

                      <div>
                        <div className="text-sm text-muted-foreground">Win Probabilities</div>
                        <div className="flex gap-2">
                          <Badge variant="outline">
                            {getTeamName(item.prediction.homeTeam)}: {(item.prediction.winProbHome * 100).toFixed(1)}%
                          </Badge>
                          <Badge variant="outline">
                            {getTeamName(item.prediction.awayTeam)}: {(item.prediction.winProbAway * 100).toFixed(1)}%
                          </Badge>
                        </div>
                      </div>
                    </div>

                    <div className="mt-4 pt-4 border-t">
                      <div className="text-sm text-muted-foreground">
                        Predicted on {new Date(item.createdAt).toLocaleString()}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        )}
      </main>
    </div>
  )
}
