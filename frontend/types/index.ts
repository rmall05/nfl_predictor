import type React from "react"
// NFL Team interface
export interface NFLTeam {
  id: string
  name: string
  city: string
  abbreviation: string
  primaryColor: string
  secondaryColor: string
  conference: "AFC" | "NFC"
  division: "North" | "South" | "East" | "West"
}

// Game prediction response
export interface PredictionResponse {
  teamA_win_prob: number
  teamB_win_prob: number
  confidence: number
  key_factors: Array<{
    name: string
    value: number
    description: string
  }>
}

// Team stats response
export interface TeamStats {
  name: string
  record: string
  epa_per_play: number
  success_rate: number
  recent_form: Array<"W" | "L">
  momentum_score: number
}

// Model performance response
export interface ModelPerformance {
  test_accuracy: number
  roc_auc: number
  generalization_gap: number
  feature_count: number
  historical_accuracy: Array<{ year: number; accuracy: number }>
  feature_importance: Array<{ feature: string; importance: number }>
}

// Component prop interfaces
export interface TeamSelectorProps {
  selectedTeam: string
  onTeamChange: (team: string) => void
  teams: NFLTeam[]
  label: string
}

export interface PredictionProps {
  teamAProb: number
  teamBProb: number
  confidence: number
  factors: Array<{ name: string; value: number; description: string }>
}

export interface ChartProps {
  data: Array<{ year: number; accuracy: number }>
  title: string
  type: "line" | "bar" | "heatmap"
}

export interface MetricCardProps {
  title: string
  value: string | number
  subtitle?: string
  trend?: "up" | "down" | "neutral"
  icon?: React.ReactNode
}
