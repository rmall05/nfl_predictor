import type {
  PredictRequest,
  PredictResponse,
  Team,
  PCAInfo,
  PCALoadings,
  PerformanceMetrics,
  FeatureImportanceResponse,
  APIError,
} from "./types"

class APIClient {
  private baseURL: string
  private cache = new Map<string, { data: any; timestamp: number }>()
  private readonly CACHE_TTL = 5 * 60 * 1000 // 5 minutes

  constructor() {
    this.baseURL = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000"
  }

  private async request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const url = `${this.baseURL}${endpoint}`
    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), 10000) // 10s timeout

    try {
      const response = await fetch(url, {
        ...options,
        signal: controller.signal,
        headers: {
          "Content-Type": "application/json",
          ...options.headers,
        },
      })

      clearTimeout(timeoutId)

      if (!response.ok) {
        const errorData: APIError = await response.json()
        throw new Error(errorData.error.message || "API request failed")
      }

      return await response.json()
    } catch (error) {
      clearTimeout(timeoutId)
      if (error instanceof Error && error.name === "AbortError") {
        throw new Error("Request timeout")
      }
      throw error
    }
  }

  private getCached<T>(key: string): T | null {
    const cached = this.cache.get(key)
    if (cached && Date.now() - cached.timestamp < this.CACHE_TTL) {
      return cached.data
    }
    return null
  }

  private setCache(key: string, data: any): void {
    this.cache.set(key, { data, timestamp: Date.now() })
  }

  async predict(request: PredictRequest): Promise<PredictResponse> {
    return this.request<PredictResponse>("/predict", {
      method: "POST",
      body: JSON.stringify(request),
    })
  }

  async getTeams(): Promise<Team[]> {
    const cacheKey = "teams"
    const cached = this.getCached<Team[]>(cacheKey)
    if (cached) return cached

    const teams = await this.request<Team[]>("/teams")
    this.setCache(cacheKey, teams)
    return teams
  }

  async getPCAInfo(): Promise<PCAInfo> {
    const cacheKey = "pca-info"
    const cached = this.getCached<PCAInfo>(cacheKey)
    if (cached) return cached

    const data = await this.request<PCAInfo>("/analytics/pca/info")
    this.setCache(cacheKey, data)
    return data
  }

  async getPCALoadings(): Promise<PCALoadings> {
    const cacheKey = "pca-loadings"
    const cached = this.getCached<PCALoadings>(cacheKey)
    if (cached) return cached

    const data = await this.request<PCALoadings>("/analytics/pca/loadings")
    this.setCache(cacheKey, data)
    return data
  }

  async getPerformanceMetrics(): Promise<PerformanceMetrics> {
    const cacheKey = "performance"
    const cached = this.getCached<PerformanceMetrics>(cacheKey)
    if (cached) return cached

    const data = await this.request<PerformanceMetrics>("/analytics/performance")
    this.setCache(cacheKey, data)
    return data
  }

  async getFeatureImportance(): Promise<FeatureImportanceResponse> {
    const cacheKey = "feature-importance"
    const cached = this.getCached<FeatureImportanceResponse>(cacheKey)
    if (cached) return cached

    const data = await this.request<FeatureImportanceResponse>("/analytics/feature-importance")
    this.setCache(cacheKey, data)
    return data
  }

  clearCache(): void {
    this.cache.clear()
  }
}

export const apiClient = new APIClient()
