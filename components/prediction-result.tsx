"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { Copy, Share2 } from "lucide-react"
import type { PredictResponse, Team } from "@/lib/types"
import { mockTeams } from "@/lib/mock-data"
import { useToast } from "@/hooks/use-toast"

interface PredictionResultProps {
  prediction: PredictResponse
}

export function PredictionResult({ prediction }: PredictionResultProps) {
  const { toast } = useToast()

  const getTeamInfo = (abbr: string): Team | undefined => {
    return mockTeams.find((team) => team.abbr === abbr)
  }

  const homeTeam = getTeamInfo(prediction.homeTeam)
  const awayTeam = getTeamInfo(prediction.awayTeam)

  const copyToClipboard = () => {
    const text = `NFL Prediction: ${awayTeam?.name} @ ${homeTeam?.name}
Winner: ${prediction.predictedWinner === "home" ? homeTeam?.name : awayTeam?.name}
Margin: ${Math.abs(prediction.predictedMargin).toFixed(1)} points
Home Win Probability: ${(prediction.winProbHome * 100).toFixed(1)}%
Away Win Probability: ${(prediction.winProbAway * 100).toFixed(1)}%`

    navigator.clipboard.writeText(text)
    toast({
      title: "Copied to clipboard",
      description: "Prediction summary has been copied to your clipboard.",
    })
  }

  const shareResult = () => {
    if (navigator.share) {
      navigator.share({
        title: "NFL Game Prediction",
        text: `${awayTeam?.name} @ ${homeTeam?.name} - Predicted winner: ${prediction.predictedWinner === "home" ? homeTeam?.name : awayTeam?.name}`,
        url: window.location.href,
      })
    } else {
      copyToClipboard()
    }
  }

  return (
    <div className="space-y-6">
      {/* Main Prediction Card */}
      <Card>
        <CardHeader>
          <CardTitle className="text-center text-2xl">
            {awayTeam?.name} @ {homeTeam?.name}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Winner and Margin */}
          <div className="text-center space-y-2">
            <div className="text-3xl font-bold text-primary">
              Predicted Winner: {prediction.predictedWinner === "home" ? homeTeam?.name : awayTeam?.name}
            </div>
            <div className="text-xl text-muted-foreground">
              Margin: {Math.abs(prediction.predictedMargin).toFixed(1)} points
            </div>
          </div>

          {/* Win Probabilities */}
          <div className="space-y-4">
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <span className="font-medium">{homeTeam?.name} (Home)</span>
                <span className="font-bold">{(prediction.winProbHome * 100).toFixed(1)}%</span>
              </div>
              <Progress value={prediction.winProbHome * 100} className="h-3" />
            </div>

            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <span className="font-medium">{awayTeam?.name} (Away)</span>
                <span className="font-bold">{(prediction.winProbAway * 100).toFixed(1)}%</span>
              </div>
              <Progress value={prediction.winProbAway * 100} className="h-3" />
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex gap-2 justify-center">
            <Button variant="outline" onClick={copyToClipboard}>
              <Copy className="mr-2 h-4 w-4" />
              Copy
            </Button>
            <Button variant="outline" onClick={shareResult}>
              <Share2 className="mr-2 h-4 w-4" />
              Share
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Top Features */}
      <Card>
        <CardHeader>
          <CardTitle>Key Factors</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {prediction.topFeatures.map((feature, index) => (
              <div key={index} className="flex items-center justify-between">
                <span className="text-sm font-medium capitalize">{feature.name.replace(/_/g, " ")}</span>
                <Badge variant={feature.value > 0 ? "default" : "secondary"}>
                  {feature.value > 0 ? "+" : ""}
                  {feature.value.toFixed(3)}
                </Badge>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
