// popup.js
document.addEventListener('DOMContentLoaded', function() {
    // Prediction Tab Elements
    const predictBtn = document.getElementById('predictBtn');
    const symbolInput = document.getElementById('symbol');
    const timeframeSelect = document.getElementById('timeframe');
    const predictResultDiv = document.getElementById('predictResult');
    const predictLoadingDiv = document.getElementById('predictLoading');

    // Training Tab Elements
    const trainBtn = document.getElementById('trainBtn');
    const trainSymbolInput = document.getElementById('trainSymbol');
    const trainDaysSelect = document.getElementById('trainDays');
    const trainResultDiv = document.getElementById('trainResult');
    const trainLoadingDiv = document.getElementById('trainLoading');

    // Try to get symbol from TradingView URL
    chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
        if (tabs && tabs[0] && tabs[0].url) {
            const url = tabs[0].url;
            const symbolMatch = url.match(/symbol=([^&]+)/) || url.match(/symbols\/([^\/]+)/);
            if (symbolMatch && symbolMatch[1]) {
                const symbol =  symbolMatch[1].split(':').pop().replace('-', '');
                symbolInput.value = symbol;
                trainSymbolInput.value = symbol;
            }
        }
    });

    // Prediction Button Click
    predictBtn.addEventListener('click', function() {
        const symbol = symbolInput.value.trim().toUpperCase();
        const timeframe = timeframeSelect.value;
        
        if (!symbol) {
            showPredictError('Please enter a stock symbol');
            return;
        }

        predictLoadingDiv.style.display = 'block';
        predictResultDiv.innerHTML = '';

        fetch('http://localhost:5000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                symbol: symbol,
                timeframe: timeframe
            })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            predictLoadingDiv.style.display = 'none';
            
            if (data.status === 'success') {
                displayPredictionResults(data);
            } else {
                showPredictError(data.error || 'Prediction failed');
            }
        })
        .catch(error => {
            predictLoadingDiv.style.display = 'none';
            showPredictError(`Connection failed. Ensure:<br>
                            1. Prediction server is running<br>
                            2. Correct URL (http://localhost:5000)<br>
                            3. No browser restrictions`);
            console.error('Error:', error);
        });
    });

    // Train Button Click
    trainBtn.addEventListener('click', function() {
        const symbol = trainSymbolInput.value.trim().toUpperCase();
        const days = trainDaysSelect.value;
        
        if (!symbol) {
            showTrainError('Please enter a stock symbol');
            return;
        }

        trainLoadingDiv.style.display = 'block';
        trainResultDiv.innerHTML = '';

        fetch('http://localhost:5000/train', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                symbol: symbol,
                days: days
            })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            trainLoadingDiv.style.display = 'none';
            
            if (data.status === 'success') {
                showTrainSuccess(`Model trained successfully for ${symbol}<br>
                                Epochs: ${data.epochs}<br>
                                Final Loss: ${data.loss.toFixed(4)}`);
            } else {
                showTrainError(data.error || 'Training failed');
            }
        })
        .catch(error => {
            trainLoadingDiv.style.display = 'none';
            showTrainError(`Training failed. Ensure:<br>
                          1. Training server is running<br>
                          2. Python environment is properly setup<br>
                          3. Enough historical data exists`);
            console.error('Error:', error);
        });
    });

    function displayPredictionResults(data) {
        let html = `
            <h3>${data.symbol} Prediction</h3>
            <p><strong>Current Price:</strong> $${data.current_price}</p>
            <div class="prediction-item">
                <strong>5-Min Prediction:</strong> 
                <span class="${data.five_min_prediction > data.current_price ? 'positive' : 'negative'}">
                    $${data.five_min_prediction} 
                    (${data.five_min_prediction > data.current_price ? '+' : ''}${((data.five_min_prediction - data.current_price) / data.current_price * 100).toFixed(2)}%)
                </span>
            </div>
            <h4>Minute-by-Minute:</h4>
        `;

        data.minute_predictions.forEach(pred => {
            html += `
                <div class="prediction-item">
                    <strong>Minute ${pred.minute}:</strong> 
                    <span class="${pred.change >= 0 ? 'positive' : 'negative'}">
                        $${pred.price} 
                        (${pred.change >= 0 ? '+' : ''}${pred.change_pct.toFixed(2)}%)
                    </span>
                </div>
            `;
        });

        html += `<p><small>Last updated: ${new Date(data.timestamp).toLocaleTimeString()}</small></p>`;
        predictResultDiv.innerHTML = html;
    }

    function showPredictError(message) {
        predictResultDiv.innerHTML = `<div class="error">${message}</div>`;
    }document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', function() {
        openTab(this.textContent.trim() === 'Predict' ? 'predictTab' : 'trainTab');
    });
});
function openTab(tabId) {
    const contents = document.querySelectorAll('.tab-content');
    contents.forEach(c => c.style.display = 'none');

    const selectedTab = document.getElementById(tabId);
    if (selectedTab) {
        selectedTab.style.display = 'block';
    }

    document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
    if (tabId === 'predictTab') {
        document.querySelector('.tab:nth-child(1)').classList.add('active');
    } else {
        document.querySelector('.tab:nth-child(2)').classList.add('active');
    }
}

    function showTrainError(message) {
        trainResultDiv.innerHTML = `<div class="error">${message}</div>`;
    }

    function showTrainSuccess(message) {
        trainResultDiv.innerHTML = `<div class="success">${message}</div>`;
    }
});
