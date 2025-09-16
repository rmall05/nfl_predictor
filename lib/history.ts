import type { PredictionHistory, PredictResponse } from "./types"

const STORAGE_KEY = "nfl_prediction_history"
const MAX_HISTORY_ITEMS = 50

export class HistoryManager {
  static getHistory(): PredictionHistory[] {
    if (typeof window === "undefined") return []

    try {
      const stored = localStorage.getItem(STORAGE_KEY)
      return stored ? JSON.parse(stored) : []
    } catch {
      return []
    }
  }

  static addPrediction(prediction: PredictResponse): void {
    if (typeof window === "undefined") return

    const history = this.getHistory()
    const newItem: PredictionHistory = {
      id: `${prediction.gameId}_${Date.now()}`,
      prediction,
      createdAt: new Date().toISOString(),
    }

    history.unshift(newItem)

    // Keep only the most recent items
    if (history.length > MAX_HISTORY_ITEMS) {
      history.splice(MAX_HISTORY_ITEMS)
    }

    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(history))
    } catch (error) {
      console.error("Failed to save prediction to history:", error)
    }
  }

  static clearHistory(): void {
    if (typeof window === "undefined") return
    localStorage.removeItem(STORAGE_KEY)
  }

  static removePrediction(id: string): void {
    if (typeof window === "undefined") return

    const history = this.getHistory().filter((item) => item.id !== id)
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(history))
    } catch (error) {
      console.error("Failed to remove prediction from history:", error)
    }
  }
}
