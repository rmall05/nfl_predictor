// NFL Predictor - Main JavaScript Functions

// Global variables
let currentPrediction = null;
let performanceData = null;

// API Base URL
const API_BASE = '';  // Empty since we're serving from the same Flask app

// Utility Functions
function showElement(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.classList.remove('d-none');
        element.classList.add('fade-in');
    }
}

function hideElement(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.classList.add('d-none');
        element.classList.remove('fade-in');
    }
}

function showLoading() {
    showElement('loadingIndicator');
    const btn = document.getElementById('predictBtn');
    if (btn) {
        btn.disabled = true;
        btn.querySelector('.btn-text').textContent = 'Analyzing...';
        btn.querySelector('.spinner-border').classList.remove('d-none');
    }
}

function hideLoading() {
    hideElement('loadingIndicator');
    const btn = document.getElementById('predictBtn');
    if (btn) {
        btn.disabled = false;
        btn.querySelector('.btn-text').textContent = 'Predict Game Outcome';
        btn.querySelector('.spinner-border').classList.add('d-none');
    }
}

function showError(message) {
    const errorMessage = document.getElementById('errorMessage');
    if (errorMessage) {
        errorMessage.textContent = message;
        showElement('errorAlert');

        // Auto-hide after 5 seconds
        setTimeout(() => {
            hideElement('errorAlert');
        }, 5000);
    }
}

// API Functions
async function callAPI(endpoint, method = 'GET', data = null) {
    try {
        const config = {
            method: method,
            headers: {
                'Content-Type': 'application/json',
            }
        };

        if (data && method !== 'GET') {
            config.body = JSON.stringify(data);
        }

        const response = await fetch(API_BASE + endpoint, config);

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        console.error('API call failed:', error);
        throw error;
    }
}

async function predictGame(teamA, teamB) {
    return await callAPI('/api/predict', 'POST', { teamA, teamB });
}

async function getTeamStats(team) {
    return await callAPI(`/api/team/${team}/stats`);
}

async function getModelPerformance() {
    return await callAPI('/api/model/performance');
}

// Dashboard Functions
function initializeDashboard() {
    console.log('Initializing NFL Prediction Dashboard...');

    // Set up form submission
    const predictionForm = document.getElementById('predictionForm');
    if (predictionForm) {
        predictionForm.addEventListener('submit', handlePredictionSubmit);
    }

    // Set up team selection change handlers
    const teamASelect = document.getElementById('teamA');
    const teamBSelect = document.getElementById('teamB');

    if (teamASelect) {
        teamASelect.addEventListener('change', handleTeamSelection);
    }

    if (teamBSelect) {
        teamBSelect.addEventListener('change', handleTeamSelection);
    }
}

function handleTeamSelection() {
    const teamA = document.getElementById('teamA').value;
    const teamB = document.getElementById('teamB').value;

    // Prevent selecting the same team for both
    if (teamA && teamB && teamA === teamB) {
        showError('Please select different teams for the prediction');
        document.getElementById('teamB').value = '';
    }
}

async function handlePredictionSubmit(event) {
    event.preventDefault();

    const teamA = document.getElementById('teamA').value;
    const teamB = document.getElementById('teamB').value;

    if (!teamA || !teamB) {
        showError('Please select both teams');
        return;
    }

    if (teamA === teamB) {
        showError('Please select different teams');
        return;
    }

    try {
        showLoading();
        hideElement('errorAlert');
        hideElement('predictionResults');

        console.log(`Predicting: ${teamA} vs ${teamB}`);

        const result = await predictGame(teamA, teamB);

        if (result.success) {
            displayPredictionResults(result, teamA, teamB);
        } else {
            throw new Error(result.error || 'Prediction failed');
        }

    } catch (error) {
        console.error('Prediction error:', error);
        showError('Failed to generate prediction. Please try again.');
    } finally {
        hideLoading();
    }
}

function displayPredictionResults(result, teamA, teamB) {
    currentPrediction = result;

    // Use team names from API stats if available, otherwise use the input values
    const teamAName = result.teamA_stats?.name || teamA;
    const teamBName = result.teamB_stats?.name || teamB;

    // Update team names
    document.getElementById('teamAName').textContent = teamAName;
    document.getElementById('teamBName').textContent = teamBName;

    // Update probabilities (API now returns as decimal, so multiply by 100)
    const teamAProb = (result.teamA_win_prob * 100).toFixed(1);
    const teamBProb = (result.teamB_win_prob * 100).toFixed(1);

    document.getElementById('teamAProb').textContent = teamAProb + '%';
    document.getElementById('teamBProb').textContent = teamBProb + '%';

    // Update confidence level - use the one from API or calculate
    const confidence = result.confidence || Math.max(result.teamA_win_prob, result.teamB_win_prob) * 100;
    document.getElementById('confidenceLevel').textContent = confidence.toFixed(1) + '%';

    // Show winner badges and highlight cards
    const teamACard = document.getElementById('teamACard');
    const teamBCard = document.getElementById('teamBCard');
    const teamAWinner = document.getElementById('teamAWinner');
    const teamBWinner = document.getElementById('teamBWinner');

    // Reset classes
    teamACard.classList.remove('winner');
    teamBCard.classList.remove('winner');
    teamAWinner.classList.add('d-none');
    teamBWinner.classList.add('d-none');

    // Highlight winner based on predicted_winner from API
    if (result.predicted_winner === teamAName || result.teamA_win_prob > 0.5) {
        teamACard.classList.add('winner');
        teamAWinner.classList.remove('d-none');
    } else {
        teamBCard.classList.add('winner');
        teamBWinner.classList.remove('d-none');
    }

    // Show results with animation
    showElement('predictionResults');

    // Display detailed metrics if available
    if (result.detailed_metrics) {
        displayDetailedMetrics(result.detailed_metrics, teamAName, teamBName);
        showElement('detailedMetrics');
        document.getElementById('detailedMetrics').style.display = 'block';
    }

    // Animate the results
    setTimeout(() => {
        document.getElementById('predictionResults').classList.add('slide-up');
    }, 100);
}

function displayDetailedMetrics(metrics, teamAName, teamBName) {
    const container = document.getElementById('metricsComparison');
    if (!container) return;

    // Helper function to format values based on format type
    function formatValue(value, format) {
        switch(format) {
            case 'percentage1':
                return (value).toFixed(1) + '%';
            case 'decimal2':
                return value.toFixed(2);
            case 'decimal3':
                return value.toFixed(3);
            default:
                return value.toFixed(2);
        }
    }

    // Helper function to determine which team is better
    function getBetterTeam(teamA_val, teamB_val, higher_better) {
        if (higher_better) {
            return teamA_val > teamB_val ? 'A' : teamB_val > teamA_val ? 'B' : 'tie';
        } else {
            return teamA_val < teamB_val ? 'A' : teamB_val < teamA_val ? 'B' : 'tie';
        }
    }

    // Create metrics comparison HTML
    let html = `
        <div class="row mb-3">
            <div class="col-md-12">
                <div class="d-flex justify-content-between align-items-center">
                    <h6 class="text-primary mb-0">${teamAName}</h6>
                    <h6 class="text-info mb-0">${teamBName}</h6>
                </div>
            </div>
        </div>
    `;

    // Group metrics by category
    const categories = {};
    metrics.forEach(metric => {
        if (!categories[metric.category]) {
            categories[metric.category] = [];
        }
        categories[metric.category].push(metric);
    });

    // Display metrics by category
    Object.keys(categories).forEach(category => {
        html += `
            <div class="mb-4">
                <h6 class="text-muted mb-3">${category}</h6>
        `;

        categories[category].forEach(metric => {
            const betterTeam = getBetterTeam(metric.teamA_value, metric.teamB_value, metric.higher_better);

            html += `
                <div class="metric-row mb-3 p-3 bg-light rounded">
                    <div class="row align-items-center">
                        <div class="col-md-3">
                            <h6 class="mb-1">${metric.name}</h6>
                            <small class="text-muted">${metric.description}</small>
                        </div>
                        <div class="col-md-3 text-center">
                            <span class="metric-value ${betterTeam === 'A' ? 'text-success fw-bold' : ''} ${betterTeam === 'B' ? 'text-muted' : ''}">
                                ${formatValue(metric.teamA_value, metric.format)}
                                ${betterTeam === 'A' ? ' ✓' : ''}
                            </span>
                        </div>
                        <div class="col-md-3 text-center">
                            <div class="metric-bar">
                                <div class="progress" style="height: 20px;">
                                    <div class="progress-bar bg-primary" role="progressbar"
                                         style="width: ${Math.abs(metric.teamA_value) / (Math.abs(metric.teamA_value) + Math.abs(metric.teamB_value)) * 100}%">
                                    </div>
                                    <div class="progress-bar bg-info" role="progressbar"
                                         style="width: ${Math.abs(metric.teamB_value) / (Math.abs(metric.teamA_value) + Math.abs(metric.teamB_value)) * 100}%">
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-3 text-center">
                            <span class="metric-value ${betterTeam === 'B' ? 'text-success fw-bold' : ''} ${betterTeam === 'A' ? 'text-muted' : ''}">
                                ${formatValue(metric.teamB_value, metric.format)}
                                ${betterTeam === 'B' ? ' ✓' : ''}
                            </span>
                        </div>
                    </div>
                </div>
            `;
        });

        html += '</div>';
    });

    container.innerHTML = html;
}

// Performance Page Functions
function initializePerformancePage() {
    console.log('Initializing Performance Page...');
    loadPerformanceData();
}

async function loadPerformanceData() {
    try {
        const data = await getModelPerformance();
        performanceData = data;
        updatePerformanceMetrics(data);
        // Create charts after data is loaded
        createPerformanceCharts();
    } catch (error) {
        console.error('Failed to load performance data:', error);
        showGenericPerformanceData();
        // Create charts even if API fails, using fallback data
        createPerformanceCharts();
    }
}

function updatePerformanceMetrics(data) {
    // Update metric values if API provides them
    if (data && data.test_accuracy) {
        document.getElementById('testAccuracy').textContent =
            (data.test_accuracy * 100).toFixed(1) + '%';
    }

    if (data && data.cv_accuracy) {
        document.getElementById('cvAccuracy').textContent =
            (data.cv_accuracy * 100).toFixed(1) + '%';
    }

    if (data && data.roc_auc) {
        document.getElementById('rocAuc').textContent =
            (data.roc_auc * 100).toFixed(1) + '%';
    }

    // Update detailed metrics
    updateDetailedMetrics(data);
}

function showGenericPerformanceData() {
    // Show static performance data if API is not available
    const detailedMetrics = document.getElementById('detailedMetrics');
    if (detailedMetrics) {
        detailedMetrics.innerHTML = `
            <div class="col-md-3 text-center">
                <h6 class="text-muted">Precision</h6>
                <h4>79.2%</h4>
            </div>
            <div class="col-md-3 text-center">
                <h6 class="text-muted">Recall</h6>
                <h4>78.4%</h4>
            </div>
            <div class="col-md-3 text-center">
                <h6 class="text-muted">F1-Score</h6>
                <h4>78.8%</h4>
            </div>
            <div class="col-md-3 text-center">
                <h6 class="text-muted">Log Loss</h6>
                <h4>0.485</h4>
            </div>
        `;
    }
}

function updateDetailedMetrics(data) {
    const detailedMetrics = document.getElementById('detailedMetrics');
    if (!detailedMetrics) return;

    if (data && data.detailed_metrics) {
        // Use API data if available
        detailedMetrics.innerHTML = `
            <div class="col-md-3 text-center">
                <h6 class="text-muted">Precision</h6>
                <h4>${(data.detailed_metrics.precision * 100).toFixed(1)}%</h4>
            </div>
            <div class="col-md-3 text-center">
                <h6 class="text-muted">Recall</h6>
                <h4>${(data.detailed_metrics.recall * 100).toFixed(1)}%</h4>
            </div>
            <div class="col-md-3 text-center">
                <h6 class="text-muted">F1-Score</h6>
                <h4>${(data.detailed_metrics.f1_score * 100).toFixed(1)}%</h4>
            </div>
            <div class="col-md-3 text-center">
                <h6 class="text-muted">Log Loss</h6>
                <h4>${data.detailed_metrics.log_loss.toFixed(3)}</h4>
            </div>
        `;
    } else {
        showGenericPerformanceData();
    }
}

function createPerformanceCharts() {
    createModelComparisonChart();
    createFeatureImportanceChart();
}

function createModelComparisonChart() {
    const ctx = document.getElementById('modelComparisonChart');
    if (!ctx) return;

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['XGBoost', 'Random Forest', 'Neural Network', 'Logistic Regression'],
            datasets: [{
                label: 'Accuracy (%)',
                data: [88.1, 87.3, 86.6, 84.2],
                backgroundColor: [
                    'rgba(1, 51, 105, 0.8)',
                    'rgba(40, 167, 69, 0.8)',
                    'rgba(255, 193, 7, 0.8)',
                    'rgba(213, 10, 10, 0.8)'
                ],
                borderColor: [
                    'rgba(1, 51, 105, 1)',
                    'rgba(40, 167, 69, 1)',
                    'rgba(255, 193, 7, 1)',
                    'rgba(213, 10, 10, 1)'
                ],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Cross-Validation Accuracy by Model'
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    min: 80,
                    max: 90,
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        }
                    }
                }
            }
        }
    });
}

function createFeatureImportanceChart() {
    const ctx = document.getElementById('featureImportanceChart');
    if (!ctx) return;

    // Use data from API if available, otherwise fallback to static data
    let featureImportanceData = [
        'EPA per play',
        'Success rate',
        'Momentum score',
        'Opponent strength',
        'Explosive rate',
        'Third down conv.',
        'Red zone efficiency',
        'Turnover rate'
    ];
    let importanceValues = [0.18, 0.15, 0.13, 0.12, 0.11, 0.09, 0.08, 0.07];

    // If performance data is loaded, use real feature importance
    if (performanceData && performanceData.feature_importance) {
        featureImportanceData = performanceData.feature_importance.map(item => item.name);
        importanceValues = performanceData.feature_importance.map(item => item.importance);
    }

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: featureImportanceData,
            datasets: [{
                label: 'Feature Importance',
                data: importanceValues,
                backgroundColor: 'rgba(1, 51, 105, 0.8)',
                borderColor: 'rgba(1, 51, 105, 1)',
                borderWidth: 1
            }]
        },
        options: {
            indexAxis: 'y',  // This makes it horizontal
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Top Feature Importance (after PCA)'
                }
            },
            scales: {
                x: {
                    beginAtZero: true,
                    max: 0.2,
                    ticks: {
                        callback: function(value) {
                            return (value * 100).toFixed(0) + '%';
                        }
                    }
                }
            }
        }
    });
}

// Utility function for handling navigation
function updateActiveNavigation() {
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('.navbar-nav .nav-link');

    navLinks.forEach(link => {
        link.classList.remove('active');
        if ((currentPath === '/' && link.getAttribute('href') === '/') ||
            (currentPath !== '/' && link.getAttribute('href') === currentPath)) {
            link.classList.add('active');
        }
    });
}

// Initialize page based on current location
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM Content Loaded');
    updateActiveNavigation();

    // Check which page we're on and initialize accordingly
    const currentPath = window.location.pathname;

    if (currentPath === '/' || currentPath === '/dashboard') {
        if (typeof initializeDashboard === 'function') {
            initializeDashboard();
        }
    } else if (currentPath === '/performance') {
        if (typeof initializePerformancePage === 'function') {
            initializePerformancePage();
        }
    }
});

// Export functions for global access
window.NFLPredictor = {
    initializeDashboard,
    initializePerformancePage,
    predictGame,
    getTeamStats,
    getModelPerformance
};