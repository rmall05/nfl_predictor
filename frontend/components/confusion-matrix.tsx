"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

// Mock confusion matrix data
const confusionData = [
  [1247, 312], // Actual Win: [Predicted Win, Predicted Loss]
  [289, 1152], // Actual Loss: [Predicted Win, Predicted Loss]
]

const labels = ["Win", "Loss"]

export function ConfusionMatrix() {
  const total = confusionData.flat().reduce((sum, val) => sum + val, 0)

  const getIntensity = (value: number) => {
    const max = Math.max(...confusionData.flat())
    const intensity = value / max
    return intensity
  }

  const getAccuracy = () => {
    const correct = confusionData[0][0] + confusionData[1][1]
    return ((correct / total) * 100).toFixed(1)
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-lg">Confusion Matrix</CardTitle>
        <p className="text-sm text-muted-foreground">Overall Accuracy: {getAccuracy()}%</p>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {/* Matrix */}
          <div className="grid grid-cols-3 gap-2 max-w-xs mx-auto">
            {/* Header row */}
            <div></div>
            <div className="text-center text-sm font-medium text-muted-foreground">Predicted Win</div>
            <div className="text-center text-sm font-medium text-muted-foreground">Predicted Loss</div>

            {/* Data rows */}
            {confusionData.map((row, i) => (
              <>
                <div key={`label-${i}`} className="flex items-center text-sm font-medium text-muted-foreground">
                  Actual {labels[i]}
                </div>
                {row.map((value, j) => {
                  const intensity = getIntensity(value)
                  const isCorrect = i === j
                  return (
                    <div
                      key={`cell-${i}-${j}`}
                      className={`
                        aspect-square flex items-center justify-center rounded-lg text-sm font-bold
                        ${isCorrect ? "bg-green-500 text-white" : "bg-red-500 text-white"}
                      `}
                      style={{
                        opacity: 0.6 + intensity * 0.4,
                      }}
                    >
                      {value}
                    </div>
                  )
                })}
              </>
            ))}
          </div>

          {/* Metrics */}
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div className="space-y-1">
              <p className="font-medium">Precision (Win)</p>
              <p className="text-muted-foreground">
                {((confusionData[0][0] / (confusionData[0][0] + confusionData[1][0])) * 100).toFixed(1)}%
              </p>
            </div>
            <div className="space-y-1">
              <p className="font-medium">Recall (Win)</p>
              <p className="text-muted-foreground">
                {((confusionData[0][0] / (confusionData[0][0] + confusionData[0][1])) * 100).toFixed(1)}%
              </p>
            </div>
            <div className="space-y-1">
              <p className="font-medium">Precision (Loss)</p>
              <p className="text-muted-foreground">
                {((confusionData[1][1] / (confusionData[1][1] + confusionData[0][1])) * 100).toFixed(1)}%
              </p>
            </div>
            <div className="space-y-1">
              <p className="font-medium">Recall (Loss)</p>
              <p className="text-muted-foreground">
                {((confusionData[1][1] / (confusionData[1][1] + confusionData[1][0])) * 100).toFixed(1)}%
              </p>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
