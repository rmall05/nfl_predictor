// API Types for NFL Prediction System

export interface PredictRequest {
  season: number
  week: number
  homeTeam: string
  awayTeam: string
  includeMomentum: boolean
}

export interface PredictResponse {
  gameId: string
  homeTeam: string
  awayTeam: string
  winProbHome: number
  winProbAway: number
  predictedWinner: "home" | "away"
  predictedMargin: number
  topFeatures: Array<{
    name: string
    value: number
  }>
  timestamp: string
}

export interface Team {
  id: string
  name: string
  abbr: string
  color?: string
}

export interface PCAInfo {
  nComponents: number
  explainedVarianceRatio: number[]
  cumulativeVarianceRatio: number[]
}

export interface PCALoadings {
  components: number[][]
  featureNames: string[]
}

export interface PerformanceMetrics {
  accuracy: number
  rocAuc: number
  precision?: number
  recall?: number
  f1?: number
  confusionMatrix: number[][]
}

export interface FeatureImportanceResponse {
  items: Array<{
    feature: string
    importance: number
  }>
}

export interface APIError {
  error: {
    code: string
    message: string
    details?: Record<string, any>
    traceId?: string
  }
}

export interface PredictionHistory {
  id: string
  prediction: PredictResponse
  createdAt: string
}
