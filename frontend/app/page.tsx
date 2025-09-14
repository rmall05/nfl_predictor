"use client"

import { useState, useEffect } from "react"
import { Navigation } from "@/components/navigation"
import { TeamSelector } from "@/components/team-selector"
import { PredictionDisplay } from "@/components/prediction-display"
import { TeamComparison } from "@/components/team-comparison"
import { NFL_TEAMS } from "@/lib/nfl-data"
import { apiClient, type PredictionResponse } from "@/lib/api-client"

export default function PredictionDashboard() {
  const [teamA, setTeamA] = useState("kc")
  const [teamB, setTeamB] = useState("buf")
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const selectedTeamA = NFL_TEAMS.find((team) => team.id === teamA)
  const selectedTeamB = NFL_TEAMS.find((team) => team.id === teamB)

  // Function to fetch prediction
  const fetchPrediction = async (teamAId: string, teamBId: string) => {
    setIsLoading(true)
    setError(null)

    try {
      const result = await apiClient.predictGame(teamAId, teamBId)
      setPrediction(result)
    } catch (err) {
      console.error('Prediction error:', err)
      setError(err instanceof Error ? err.message : 'Failed to get prediction')
      // Fallback to mock data on error
      setPrediction({
        teamA_win_prob: 65.2,
        teamB_win_prob: 34.8,
        confidence: 75,
        key_factors: [
          { name: "EPA per Play", value: 0.15, description: "Expected Points Added per play differential" },
          { name: "Momentum Score", value: 85, description: "Recent performance and trend analysis" },
          { name: "Recent Form", value: 80, description: "Performance in last 5 games" },
        ],
        teamA_stats: {
          name: selectedTeamA?.name || "Team A",
          record: "8-6",
          epa_per_play: 0.12,
          success_rate: 45.0,
          recent_form: ["W", "L", "W", "W", "L"] as Array<"W" | "L">,
          momentum_score: 75
        },
        teamB_stats: {
          name: selectedTeamB?.name || "Team B",
          record: "7-7",
          epa_per_play: 0.10,
          success_rate: 43.0,
          recent_form: ["L", "W", "L", "W", "W"] as Array<"W" | "L">,
          momentum_score: 70
        }
      })
    } finally {
      setIsLoading(false)
    }
  }

  // Fetch prediction when teams change
  useEffect(() => {
    if (teamA && teamB && teamA !== teamB) {
      fetchPrediction(teamA, teamB)
    }
  }, [teamA, teamB])

  return (
    <div className="min-h-screen bg-background">
      <Navigation />

      <main className="container mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left Column - Team Selection */}
          <div className="space-y-6">
            <div className="bg-card rounded-lg border p-6 shadow-sm">
              <h2 className="text-lg font-semibold mb-4 text-card-foreground">Select Teams</h2>
              <div className="space-y-4">
                <TeamSelector selectedTeam={teamA} onTeamChange={setTeamA} teams={NFL_TEAMS} label="Team A" />

                <div className="flex items-center justify-center py-2">
                  <div className="bg-primary/10 text-primary font-bold text-lg px-4 py-2 rounded-full">VS</div>
                </div>

                <TeamSelector selectedTeam={teamB} onTeamChange={setTeamB} teams={NFL_TEAMS} label="Team B" />
              </div>
            </div>
          </div>

          {/* Center Column - Prediction Results */}
          <div className="space-y-6">
            {error && (
              <div className="bg-red-50 border border-red-200 text-red-800 px-4 py-3 rounded-lg">
                <p className="text-sm font-medium">Prediction Error</p>
                <p className="text-sm">{error}</p>
                <p className="text-xs mt-1">Showing fallback data...</p>
              </div>
            )}

            <PredictionDisplay
              teamAProb={prediction?.teamA_win_prob ?? 50}
              teamBProb={prediction?.teamB_win_prob ?? 50}
              confidence={prediction?.confidence ?? 50}
              factors={prediction?.key_factors ?? []}
              teamAName={selectedTeamA?.name || "Team A"}
              teamBName={selectedTeamB?.name || "Team B"}
              isLoading={isLoading}
            />
          </div>

          {/* Right Column - Team Comparison */}
          <div className="space-y-6">
            <TeamComparison
              teamAStats={prediction?.teamA_stats ?? {
                name: selectedTeamA?.name || "Team A",
                record: "0-0",
                epa_per_play: 0,
                success_rate: 0,
                recent_form: [] as Array<"W" | "L">,
                momentum_score: 0
              }}
              teamBStats={prediction?.teamB_stats ?? {
                name: selectedTeamB?.name || "Team B",
                record: "0-0",
                epa_per_play: 0,
                success_rate: 0,
                recent_form: [] as Array<"W" | "L">,
                momentum_score: 0
              }}
              isLoading={isLoading}
            />
          </div>
        </div>
      </main>
    </div>
  )
}
