import type {
  Team,
  PredictResponse,
  PCAInfo,
  PCALoadings,
  PerformanceMetrics,
  FeatureImportanceResponse,
} from "./types"

export const mockTeams: Team[] = [
  { id: "BUF", name: "Buffalo Bills", abbr: "BUF", color: "#00338D" },
  { id: "MIA", name: "Miami Dolphins", abbr: "MIA", color: "#008E97" },
  { id: "NE", name: "New England Patriots", abbr: "NE", color: "#002244" },
  { id: "NYJ", name: "New York Jets", abbr: "NYJ", color: "#125740" },
  { id: "BAL", name: "Baltimore Ravens", abbr: "BAL", color: "#241773" },
  { id: "CIN", name: "Cincinnati Bengals", abbr: "CIN", color: "#FB4F14" },
  { id: "CLE", name: "Cleveland Browns", abbr: "CLE", color: "#311D00" },
  { id: "PIT", name: "Pittsburgh Steelers", abbr: "PIT", color: "#FFB612" },
  { id: "HOU", name: "Houston Texans", abbr: "HOU", color: "#03202F" },
  { id: "IND", name: "Indianapolis Colts", abbr: "IND", color: "#002C5F" },
  { id: "JAX", name: "Jacksonville Jaguars", abbr: "JAX", color: "#006778" },
  { id: "TEN", name: "Tennessee Titans", abbr: "TEN", color: "#0C2340" },
  { id: "DEN", name: "Denver Broncos", abbr: "DEN", color: "#FB4F14" },
  { id: "KC", name: "Kansas City Chiefs", abbr: "KC", color: "#E31837" },
  { id: "LV", name: "Las Vegas Raiders", abbr: "LV", color: "#000000" },
  { id: "LAC", name: "Los Angeles Chargers", abbr: "LAC", color: "#0080C6" },
  { id: "DAL", name: "Dallas Cowboys", abbr: "DAL", color: "#003594" },
  { id: "NYG", name: "New York Giants", abbr: "NYG", color: "#0B2265" },
  { id: "PHI", name: "Philadelphia Eagles", abbr: "PHI", color: "#004C54" },
  { id: "WAS", name: "Washington Commanders", abbr: "WAS", color: "#5A1414" },
  { id: "CHI", name: "Chicago Bears", abbr: "CHI", color: "#0B162A" },
  { id: "DET", name: "Detroit Lions", abbr: "DET", color: "#0076B6" },
  { id: "GB", name: "Green Bay Packers", abbr: "GB", color: "#203731" },
  { id: "MIN", name: "Minnesota Vikings", abbr: "MIN", color: "#4F2683" },
  { id: "ATL", name: "Atlanta Falcons", abbr: "ATL", color: "#A71930" },
  { id: "CAR", name: "Carolina Panthers", abbr: "CAR", color: "#0085CA" },
  { id: "NO", name: "New Orleans Saints", abbr: "NO", color: "#D3BC8D" },
  { id: "TB", name: "Tampa Bay Buccaneers", abbr: "TB", color: "#D50A0A" },
  { id: "ARI", name: "Arizona Cardinals", abbr: "ARI", color: "#97233F" },
  { id: "LAR", name: "Los Angeles Rams", abbr: "LAR", color: "#003594" },
  { id: "SF", name: "San Francisco 49ers", abbr: "SF", color: "#AA0000" },
  { id: "SEA", name: "Seattle Seahawks", abbr: "SEA", color: "#002244" },
]

export const mockPrediction: PredictResponse = {
  gameId: "2024_05_BUF_MIA",
  homeTeam: "BUF",
  awayTeam: "MIA",
  winProbHome: 0.63,
  winProbAway: 0.37,
  predictedWinner: "home",
  predictedMargin: 4.8,
  topFeatures: [
    { name: "off_epa_per_play", value: 0.22 },
    { name: "opp_allowed_success_rate", value: -0.15 },
    { name: "def_allowed_epa_per_play", value: -0.12 },
    { name: "off_success_rate", value: 0.18 },
    { name: "turnover_differential", value: 0.08 },
  ],
  timestamp: new Date().toISOString(),
}

export const mockPCAInfo: PCAInfo = {
  nComponents: 8,
  explainedVarianceRatio: [0.32, 0.18, 0.11, 0.09, 0.07, 0.06, 0.05, 0.04],
  cumulativeVarianceRatio: [0.32, 0.5, 0.61, 0.7, 0.77, 0.83, 0.88, 0.92],
}

export const mockPCALoadings: PCALoadings = {
  components: [
    [0.12, -0.07, 0.15, -0.09, 0.11, -0.08, 0.13, -0.06],
    [-0.08, 0.14, -0.11, 0.16, -0.09, 0.12, -0.07, 0.1],
    [0.09, -0.12, 0.08, -0.14, 0.1, -0.11, 0.09, -0.13],
  ],
  featureNames: [
    "off_epa_per_play",
    "off_success_rate",
    "def_allowed_epa_per_play",
    "def_allowed_success_rate",
    "turnover_differential",
    "penalty_yards_per_game",
    "red_zone_efficiency",
    "third_down_conversion",
  ],
}

export const mockPerformanceMetrics: PerformanceMetrics = {
  accuracy: 0.65,
  rocAuc: 0.69,
  precision: 0.66,
  recall: 0.63,
  f1: 0.645,
  confusionMatrix: [
    [520, 280],
    [260, 540],
  ],
}

export const mockFeatureImportance: FeatureImportanceResponse = {
  items: [
    { feature: "off_epa_per_play", importance: 0.19 },
    { feature: "def_allowed_epa_per_play", importance: 0.16 },
    { feature: "off_success_rate", importance: 0.14 },
    { feature: "turnover_differential", importance: 0.12 },
    { feature: "red_zone_efficiency", importance: 0.11 },
    { feature: "third_down_conversion", importance: 0.1 },
    { feature: "penalty_yards_per_game", importance: 0.09 },
    { feature: "def_allowed_success_rate", importance: 0.09 },
  ],
}
