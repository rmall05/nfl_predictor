"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { TrendingUp, TrendingDown, Minus } from "lucide-react"
import { motion } from "framer-motion"
import type { TeamStats } from "@/types"

interface TeamComparisonProps {
  teamAStats: TeamStats
  teamBStats: TeamStats
  isLoading?: boolean
}

export function TeamComparison({ teamAStats, teamBStats, isLoading = false }: TeamComparisonProps) {
  const getFormColor = (result: "W" | "L") => {
    return result === "W" ? "bg-green-500" : "bg-red-500"
  }

  const getTrendIcon = (valueA: number, valueB: number) => {
    if (valueA > valueB) return <TrendingUp className="h-4 w-4 text-green-500" />
    if (valueA < valueB) return <TrendingDown className="h-4 w-4 text-red-500" />
    return <Minus className="h-4 w-4 text-muted-foreground" />
  }

  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.5, delay: 0.2 }}
    >
      <Card>
        <CardHeader>
          <CardTitle>Team Comparison</CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Team Headers */}
          <motion.div
            className="grid grid-cols-2 gap-4"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3, duration: 0.5 }}
          >
            <motion.div
              className="text-center p-4 bg-primary/5 rounded-lg"
              whileHover={{ scale: 1.02 }}
              transition={{ type: "spring", stiffness: 300 }}
            >
              <h3 className="font-semibold text-primary">{teamAStats.name}</h3>
              <p className="text-2xl font-bold">{teamAStats.record}</p>
            </motion.div>
            <motion.div
              className="text-center p-4 bg-muted/50 rounded-lg"
              whileHover={{ scale: 1.02 }}
              transition={{ type: "spring", stiffness: 300 }}
            >
              <h3 className="font-semibold">{teamBStats.name}</h3>
              <p className="text-2xl font-bold">{teamBStats.record}</p>
            </motion.div>
          </motion.div>

          {/* Key Metrics */}
          <div className="space-y-4">
            {[
              {
                label: "EPA per Play",
                valueA: teamAStats.epa_per_play.toFixed(3),
                valueB: teamBStats.epa_per_play.toFixed(3),
                numA: teamAStats.epa_per_play,
                numB: teamBStats.epa_per_play,
              },
              {
                label: "Success Rate",
                valueA: `${teamAStats.success_rate.toFixed(1)}%`,
                valueB: `${teamBStats.success_rate.toFixed(1)}%`,
                numA: teamAStats.success_rate,
                numB: teamBStats.success_rate,
              },
              {
                label: "Momentum Score",
                valueA: teamAStats.momentum_score.toString(),
                valueB: teamBStats.momentum_score.toString(),
                numA: teamAStats.momentum_score,
                numB: teamBStats.momentum_score,
              },
            ].map((metric, index) => (
              <motion.div
                key={metric.label}
                className="flex items-center justify-between p-3 bg-card rounded-lg border hover:shadow-md transition-shadow"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.5 + index * 0.1, duration: 0.4 }}
                whileHover={{ scale: 1.01 }}
              >
                <span className="font-medium">{metric.label}</span>
                <div className="flex items-center space-x-4">
                  <div className="flex items-center space-x-2">
                    <span className="font-mono">{metric.valueA}</span>
                    {getTrendIcon(metric.numA, metric.numB)}
                  </div>
                  <div className="w-px h-6 bg-border" />
                  <div className="flex items-center space-x-2">
                    <span className="font-mono">{metric.valueB}</span>
                    {getTrendIcon(metric.numB, metric.numA)}
                  </div>
                </div>
              </motion.div>
            ))}
          </div>

          {/* Recent Form */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.8, duration: 0.5 }}
          >
            <h4 className="font-medium mb-3">Recent Form (Last 5 Games)</h4>
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <p className="text-sm text-muted-foreground">{teamAStats.name}</p>
                <div className="flex space-x-1">
                  {teamAStats.recent_form.map((result, index) => (
                    <motion.div
                      key={index}
                      initial={{ scale: 0, opacity: 0 }}
                      animate={{ scale: 1, opacity: 1 }}
                      transition={{ delay: 1 + index * 0.1, duration: 0.3 }}
                      whileHover={{ scale: 1.1 }}
                    >
                      <Badge
                        variant="secondary"
                        className={`${getFormColor(result)} text-white w-8 h-8 rounded-full flex items-center justify-center p-0`}
                      >
                        {result}
                      </Badge>
                    </motion.div>
                  ))}
                </div>
              </div>
              <div className="space-y-2">
                <p className="text-sm text-muted-foreground">{teamBStats.name}</p>
                <div className="flex space-x-1">
                  {teamBStats.recent_form.map((result, index) => (
                    <motion.div
                      key={index}
                      initial={{ scale: 0, opacity: 0 }}
                      animate={{ scale: 1, opacity: 1 }}
                      transition={{ delay: 1 + index * 0.1, duration: 0.3 }}
                      whileHover={{ scale: 1.1 }}
                    >
                      <Badge
                        variant="secondary"
                        className={`${getFormColor(result)} text-white w-8 h-8 rounded-full flex items-center justify-center p-0`}
                      >
                        {result}
                      </Badge>
                    </motion.div>
                  ))}
                </div>
              </div>
            </div>
          </motion.div>
        </CardContent>
      </Card>
    </motion.div>
  )
}
