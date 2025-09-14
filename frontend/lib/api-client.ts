const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000'

export interface PredictionRequest {
  teamA: string
  teamB: string
}

export interface TeamStats {
  name: string
  record: string
  epa_per_play: number
  success_rate: number
  recent_form: Array<'W' | 'L'>
  momentum_score: number
}

export interface KeyFactor {
  name: string
  value: number
  description: string
}

export interface PredictionResponse {
  teamA_win_prob: number
  teamB_win_prob: number
  confidence: number
  key_factors: KeyFactor[]
  teamA_stats: TeamStats
  teamB_stats: TeamStats
}

export class ApiClient {
  private baseUrl: string

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl
  }

  async healthCheck(): Promise<{ status: string; model_loaded: boolean }> {
    const response = await fetch(`${this.baseUrl}/api/health`)
    if (!response.ok) {
      throw new Error('API health check failed')
    }
    return response.json()
  }

  async predictGame(teamA: string, teamB: string): Promise<PredictionResponse> {
    const response = await fetch(`${this.baseUrl}/api/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ teamA, teamB }),
    })

    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.error || 'Prediction failed')
    }

    return response.json()
  }

  async getTeams(): Promise<any[]> {
    const response = await fetch(`${this.baseUrl}/api/teams`)
    if (!response.ok) {
      throw new Error('Failed to fetch teams')
    }
    return response.json()
  }
}

export const apiClient = new ApiClient()