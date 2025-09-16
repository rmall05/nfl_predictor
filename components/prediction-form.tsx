"use client"

import type React from "react"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Switch } from "@/components/ui/switch"
import { Loader2 } from "lucide-react"
import type { PredictRequest, Team } from "@/lib/types"
import { mockTeams } from "@/lib/mock-data"

interface PredictionFormProps {
  onSubmit: (request: PredictRequest) => void
  loading: boolean
}

export function PredictionForm({ onSubmit, loading }: PredictionFormProps) {
  const [teams, setTeams] = useState<Team[]>([])
  const [formData, setFormData] = useState<PredictRequest>({
    season: 2024,
    week: 1,
    homeTeam: "",
    awayTeam: "",
    includeMomentum: true,
  })
  const [errors, setErrors] = useState<Record<string, string>>({})

  useEffect(() => {
    // In a real app, this would fetch from the API
    setTeams(mockTeams)
  }, [])

  const validateForm = (): boolean => {
    const newErrors: Record<string, string> = {}

    if (!formData.homeTeam) {
      newErrors.homeTeam = "Home team is required"
    }
    if (!formData.awayTeam) {
      newErrors.awayTeam = "Away team is required"
    }
    if (formData.homeTeam === formData.awayTeam) {
      newErrors.teams = "Home and away teams must be different"
    }
    if (formData.week < 1 || formData.week > 18) {
      newErrors.week = "Week must be between 1 and 18"
    }
    if (formData.season < 2020 || formData.season > 2030) {
      newErrors.season = "Season must be between 2020 and 2030"
    }

    setErrors(newErrors)
    return Object.keys(newErrors).length === 0
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (validateForm()) {
      onSubmit(formData)
    }
  }

  return (
    <Card className="w-full max-w-2xl">
      <CardHeader>
        <CardTitle className="text-2xl font-bold text-center">NFL Game Prediction</CardTitle>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="season">Season</Label>
              <Input
                id="season"
                type="number"
                value={formData.season}
                onChange={(e) => setFormData({ ...formData, season: Number.parseInt(e.target.value) || 2024 })}
                min="2020"
                max="2030"
                className={errors.season ? "border-destructive" : ""}
              />
              {errors.season && <p className="text-sm text-destructive">{errors.season}</p>}
            </div>

            <div className="space-y-2">
              <Label htmlFor="week">Week</Label>
              <Input
                id="week"
                type="number"
                value={formData.week}
                onChange={(e) => setFormData({ ...formData, week: Number.parseInt(e.target.value) || 1 })}
                min="1"
                max="18"
                className={errors.week ? "border-destructive" : ""}
              />
              {errors.week && <p className="text-sm text-destructive">{errors.week}</p>}
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="homeTeam">Home Team</Label>
              <Select
                value={formData.homeTeam}
                onValueChange={(value) => setFormData({ ...formData, homeTeam: value })}
              >
                <SelectTrigger className={errors.homeTeam || errors.teams ? "border-destructive" : ""}>
                  <SelectValue placeholder="Select home team" />
                </SelectTrigger>
                <SelectContent>
                  {teams.map((team) => (
                    <SelectItem key={team.id} value={team.abbr}>
                      {team.name} ({team.abbr})
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              {errors.homeTeam && <p className="text-sm text-destructive">{errors.homeTeam}</p>}
            </div>

            <div className="space-y-2">
              <Label htmlFor="awayTeam">Away Team</Label>
              <Select
                value={formData.awayTeam}
                onValueChange={(value) => setFormData({ ...formData, awayTeam: value })}
              >
                <SelectTrigger className={errors.awayTeam || errors.teams ? "border-destructive" : ""}>
                  <SelectValue placeholder="Select away team" />
                </SelectTrigger>
                <SelectContent>
                  {teams.map((team) => (
                    <SelectItem key={team.id} value={team.abbr}>
                      {team.name} ({team.abbr})
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              {errors.awayTeam && <p className="text-sm text-destructive">{errors.awayTeam}</p>}
            </div>
          </div>

          {errors.teams && <p className="text-sm text-destructive text-center">{errors.teams}</p>}

          <div className="flex items-center space-x-2">
            <Switch
              id="momentum"
              checked={formData.includeMomentum}
              onCheckedChange={(checked) => setFormData({ ...formData, includeMomentum: checked })}
            />
            <Label htmlFor="momentum">Include momentum factors</Label>
          </div>

          <Button type="submit" className="w-full" disabled={loading}>
            {loading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Generating Prediction...
              </>
            ) : (
              "Get Prediction"
            )}
          </Button>
        </form>
      </CardContent>
    </Card>
  )
}
