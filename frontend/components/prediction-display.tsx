"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { Progress } from "@/components/ui/progress"
import { Info } from "lucide-react"
import { motion, AnimatePresence } from "framer-motion"
import { useState, useEffect } from "react"

interface PredictionDisplayProps {
  teamAProb: number
  teamBProb: number
  confidence: number
  factors: Array<{ name: string; value: number; description: string }>
  teamAName: string
  teamBName: string
  isLoading?: boolean
}

export function PredictionDisplay({
  teamAProb,
  teamBProb,
  confidence,
  factors,
  teamAName,
  teamBName,
  isLoading: externalLoading = false,
}: PredictionDisplayProps) {
  const [isLoading, setIsLoading] = useState(true)
  const [animatedProb, setAnimatedProb] = useState(0)

  useEffect(() => {
    // Use external loading state or simulate loading
    if (externalLoading) {
      setIsLoading(true)
    } else {
      setIsLoading(true)
      const timer = setTimeout(() => {
        setIsLoading(false)
      }, 800)
      return () => clearTimeout(timer)
    }
  }, [teamAName, teamBName, externalLoading])

  useEffect(() => {
    // Set loading to false when external loading is done
    if (!externalLoading) {
      setIsLoading(false)
    }
  }, [externalLoading])

  useEffect(() => {
    if (!isLoading) {
      // Animate the probability counter
      let start = 0
      const end = teamAProb
      const duration = 1500
      const increment = end / (duration / 16)

      const counter = setInterval(() => {
        start += increment
        if (start >= end) {
          setAnimatedProb(end)
          clearInterval(counter)
        } else {
          setAnimatedProb(start)
        }
      }, 16)

      return () => clearInterval(counter)
    }
  }, [teamAProb, isLoading])

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 80) return "bg-green-500"
    if (confidence >= 60) return "bg-yellow-500"
    return "bg-red-500"
  }

  const getConfidenceLabel = (confidence: number) => {
    if (confidence >= 80) return "High"
    if (confidence >= 60) return "Medium"
    return "Low"
  }

  if (isLoading) {
    return (
      <div className="space-y-6">
        <Card>
          <CardHeader className="text-center">
            <CardTitle className="text-2xl font-bold">Game Prediction</CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="flex items-center justify-center">
              <div className="w-48 h-48 rounded-full border-4 border-muted animate-pulse flex items-center justify-center">
                <div className="text-center">
                  <div className="w-16 h-8 bg-muted rounded animate-pulse mb-2"></div>
                  <div className="w-20 h-4 bg-muted rounded animate-pulse"></div>
                </div>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div className="p-4 bg-muted/50 rounded-lg animate-pulse">
                <div className="h-6 bg-muted rounded mb-2"></div>
                <div className="h-8 bg-muted rounded"></div>
              </div>
              <div className="p-4 bg-muted/50 rounded-lg animate-pulse">
                <div className="h-6 bg-muted rounded mb-2"></div>
                <div className="h-8 bg-muted rounded"></div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    )
  }

  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={`${teamAName}-${teamBName}`}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -20 }}
        transition={{ duration: 0.5 }}
        className="space-y-6"
      >
        {/* Win Probability Display */}
        <Card>
          <CardHeader className="text-center">
            <CardTitle className="text-2xl font-bold">Game Prediction</CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Circular Progress Gauge */}
            <div className="flex items-center justify-center">
              <div className="relative w-48 h-48">
                <svg className="w-full h-full transform -rotate-90" viewBox="0 0 100 100">
                  {/* Background circle */}
                  <circle
                    cx="50"
                    cy="50"
                    r="45"
                    stroke="currentColor"
                    strokeWidth="8"
                    fill="none"
                    className="text-muted"
                  />
                  {/* Progress circle */}
                  <motion.circle
                    cx="50"
                    cy="50"
                    r="45"
                    stroke="currentColor"
                    strokeWidth="8"
                    fill="none"
                    strokeDasharray={`${2 * Math.PI * 45}`}
                    strokeDashoffset={`${2 * Math.PI * 45 * (1 - animatedProb / 100)}`}
                    className="text-primary"
                    initial={{ strokeDashoffset: 2 * Math.PI * 45 }}
                    animate={{ strokeDashoffset: 2 * Math.PI * 45 * (1 - animatedProb / 100) }}
                    transition={{ duration: 1.5, ease: "easeOut" }}
                  />
                </svg>
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="text-center">
                    <motion.div
                      className="text-3xl font-bold text-primary"
                      initial={{ scale: 0.8 }}
                      animate={{ scale: 1 }}
                      transition={{ delay: 0.5, duration: 0.3 }}
                    >
                      {animatedProb.toFixed(1)}%
                    </motion.div>
                    <div className="text-sm text-muted-foreground">Win Probability</div>
                  </div>
                </div>
              </div>
            </div>

            {/* Team Probabilities */}
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
                <div className="text-lg font-semibold text-primary">{teamAName}</div>
                <div className="text-2xl font-bold">{teamAProb.toFixed(1)}%</div>
              </motion.div>
              <motion.div
                className="text-center p-4 bg-muted/50 rounded-lg"
                whileHover={{ scale: 1.02 }}
                transition={{ type: "spring", stiffness: 300 }}
              >
                <div className="text-lg font-semibold">{teamBName}</div>
                <div className="text-2xl font-bold">{teamBProb.toFixed(1)}%</div>
              </motion.div>
            </motion.div>

            {/* Confidence Score */}
            <motion.div
              className="flex items-center justify-center space-x-2"
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.6, duration: 0.3 }}
            >
              <Badge variant="secondary" className={`${getConfidenceColor(confidence)} text-white`}>
                {getConfidenceLabel(confidence)} Confidence: {confidence}%
              </Badge>
            </motion.div>
          </CardContent>
        </Card>

        {/* Key Factors */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <span>Key Factors</span>
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger>
                    <Info className="h-4 w-4 text-muted-foreground" />
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>Factors influencing the prediction outcome</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {factors.map((factor, index) => (
              <motion.div
                key={factor.name}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.8 + index * 0.1, duration: 0.4 }}
                className="space-y-2 group"
              >
                <div className="flex items-center justify-between">
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger className="flex items-center space-x-1 group-hover:text-primary transition-colors">
                        <span className="font-medium">{factor.name}</span>
                        <Info className="h-3 w-3 text-muted-foreground group-hover:text-primary transition-colors" />
                      </TooltipTrigger>
                      <TooltipContent>
                        <p className="max-w-xs">{factor.description}</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                  <span className="text-sm font-mono">{factor.value}</span>
                </div>
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: "100%" }}
                  transition={{ delay: 1 + index * 0.1, duration: 0.6 }}
                >
                  <Progress value={Math.abs(factor.value) * 100} className="h-2" />
                </motion.div>
              </motion.div>
            ))}
          </CardContent>
        </Card>
      </motion.div>
    </AnimatePresence>
  )
}
